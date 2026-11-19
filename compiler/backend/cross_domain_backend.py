"""Bounded two-axis recomputation for a closed pair of reduction domains.

Programs own one base row and one expanded-axis tile. Scalar base statistics
are recomputed per tile, weights per R chunk (not per D lane). Three sequential
sweeps preserve max -> sum -> weighted-sum dependencies and dtype boundaries.
"""
from compiler.backend.backends import Argument, Names, identifier, product, group_operands, group_results
from compiler.optimization.indexed_fusion import REDUCTIONS
from compiler.optimization.reduction_fusion import row_key,full_key
from compiler.backend.primitives import offset_for, tile_offset
from compiler.backend.platform_configs import emitted_index_dtype
from compiler.backend.index_coordinates import flatten, singleton_reshape, broadcast
from compiler.ir.index_maps import inverse_coordinates


def triton_kernel(graph,group,memory,name,config, *, segmentation=None, stage=None):
    from compiler.backend.triton_backend import (_input_layouts,_view_offset,_make_kernel,
        DTYPE_TO_TRITON,POINTWISE_EXPRESSION,_pointwise_expression,IndexBound)
    from compiler.backend.value_layer import NAMES as VALUE_LAYER
    operands,results=group_operands(graph,group),group_results(graph,group)
    produced={n.output:n for n in group.members};base=group.cross_domain;domain=group.reduction_domain
    width=base[-1];expanded=domain[-2];count=product(base[:-1]);base_row=row_key(base[:-1])
    reductions=[n for n in group.members if n.op in REDUCTIONS]
    wide_reduction=group.anchor
    base_reductions=[n for n in reductions if n is not wide_reduction]
    shared=next((n for n in base_reductions if any(n.output in batch and wide_reduction.output in batch
                                                  for batch in group.reduction_batches)),None)
    if segmentation is not None:
        if stage not in {'max', 'partials', 'finish', 'local', 'merge'}:
            raise ValueError('segmented lowering needs an explicit stage')
    elif stage is not None:
        raise ValueError('stage requires segmentation')
    # Base ancestors are the closed statistics subgraph. Its full values are
    # addressed in the R domain even when a wide broadcast requests them.
    base_values=set()
    def walk(v):
        if v in base_values:return
        node=produced.get(v)
        if node is None:return
        base_values.add(v)
        inputs=node.inputs[:1] if node.op in {'reshape','cast_like','cast_to','to_fp32','broadcast_like','move_first_last'} else node.inputs
        if node.op not in {'ones_like','zeros_like','neg_inf_like'}:
            for arg in inputs:walk(arg)
    for n in base_reductions:walk(n.output)
    names=Names(graph);ptrs={v:names.fresh(identifier(v)+'_ptr') for v in operands+results}
    pairs=[(ptrs[v],Argument('tensor',v)) for v in operands+results]
    dims=[]
    for v in operands+tuple(produced):
        for d in graph.values[v].shape:
            if not str(d).isdigit() and d not in dims:dims.append(d);pairs.append((f'{d}: tl.constexpr',Argument('value',str(d))))
    layouts=_input_layouts(graph,memory,operands,names,pairs)
    limit=config.reduction_block_limit or 4096
    rcap,dcap=group.cross_tile
    rt,dt=names.fresh('FG_R_TILE'),names.fresh('FG_D_TILE')
    rexpr=f'min({rcap}, triton.next_power_of_2({width}))'
    dexpr=f'min({dcap}, max(1, {limit} // ({rexpr})), triton.next_power_of_2({expanded}))'
    if segmentation is not None:
        sp,st=names.fresh('FG_SEGMENTS'),names.fresh('FG_SEGMENT_TILE')
        workptrs=tuple(names.fresh('fg_segment_ptr') for _ in segmentation.scratch)
        rexpr,dexpr=segmentation.r_tile,segmentation.d_tile
        pairs.extend((ptr,Argument('workspace',value.name)) for ptr,value in zip(workptrs,segmentation.scratch))
        pairs.extend([(f'{sp}: tl.constexpr',Argument('value',segmentation.segments)),
                      (f'{st}: tl.constexpr',Argument('value',f'triton.next_power_of_2({segmentation.segments})'))])
    pairs.extend([(f'{rt}: tl.constexpr' ,Argument('value',rexpr)),(f'{dt}: tl.constexpr',Argument('value',dexpr))])
    row,d,rv,dv=names.fresh('row'),names.fresh('d'),names.fresh('row_valid'),names.fresh('d_valid')
    # --- 32-bit program-index policy ---------------------------------------
    # Both program ids are affine in the same index space the leaves are
    # addressed in, so the bound is checkable from the host's shape/stride
    # metadata: each leaf's element count and strided extent, plus one tile of
    # masked tail lanes on the expanded axis (`d` runs `cdiv(expanded, dt)*dt`
    # lanes) and on the reduction axis. A 32-bit `d` must also not wrap before
    # its `d < expanded` mask is evaluated, hence the expanded dimension is
    # part of the count expression rather than implied by a leaf count.
    # Segmented stages address caller workspaces whose extent the host does
    # not bound here, so they stay int64.
    index_proof=None
    if segmentation is None:
        index_proof=IndexBound(
            count=f'max({expanded}, {width})',
            padding=limit,
            leaves=tuple(operands+results),
        )
    index_dtype=emitted_index_dtype(config,proven=index_proof is not None)
    index_cast='tl.int32' if index_dtype=='int32' else 'tl.int64'
    lines=[f'    {row} = tl.program_id(0).to({index_cast})',f'    {rv} = {row} < ({count})',
           f'    {d} = tl.program_id(1).to({index_cast}) * {dt} + tl.arange(0, {dt})',f'    {dv} = {d} < {expanded}']
    origin='';chunks='1' if segmentation is not None else f'tl.cdiv({width}, {rt})'
    row_cache={};cache={};base_at=None;base_mask=None;base_coordinates=None
    def emit(expr,hint,indent='    '):
        var=names.fresh(hint);lines.append(f'{indent}{var} = {expr}');return var
    # Decode the base row once. R/D coordinates already exist in each sweep.
    row_coordinates=[]
    for axis,dimension in enumerate(base[:-1]):
        tail=product(base[axis+1:-1])
        coordinate=f'({row} // ({tail}))'
        if axis:coordinate+=f' % ({dimension})'
        row_coordinates.append(emit(coordinate,'row_coordinate'))
    coordinates={}
    def load(v,at,mask,indent):
        addr_lines=[];addr=_view_offset(v,at,layouts,names,addr_lines,coordinates=coordinates.get((v,at)))
        lines.extend(indent+s[4:] for s in addr_lines)
        return emit(f'tl.load({ptrs[v]} + {addr}, mask={mask}, other=0)',identifier(v)+'_load',indent)
    def compute(v,at,mask,indent,get):
        n=produced.get(v)
        if n is None:return load(v,at,mask,indent)
        op=n.op;x=n.inputs[0];dtype=DTYPE_TO_TRITON[graph.values[v].dtype]
        known=coordinates.get((v,at))
        def remember(value,index,mapped):
            if mapped is not None:coordinates[value,index]=mapped
        if op in {'ones_like','zeros_like','neg_inf_like'}:
            body=f'tl.full((), '+{'ones_like':'1','zeros_like':'0','neg_inf_like':"float('-inf')"}[op]+f', {dtype})'
        elif op=='move_first_last':
            old=graph.values[x].shape
            inverse=inverse_coordinates(n,known) if known is not None else None
            expression=flatten(old,inverse) if inverse is not None else f'(({at}) % {old[0]}) * ({product(old[1:])}) + ({at}) // {old[0]}'
            mapped=emit(expression,'moved_index',indent)
            remember(x,mapped,inverse)
            body=get(x,mapped,mask,indent)
        elif op=='broadcast_like':
            inverse=broadcast(graph.values[x].shape,graph.values[v].shape,known) if known is not None else None
            expression=flatten(graph.values[x].shape,inverse) if inverse is not None else tile_offset(offset_for(graph.values[x].shape,graph.values[v].shape,at),at)
            mapped=emit(expression,'broadcast_index',indent)
            remember(x,mapped,inverse)
            body=get(x,mapped,mask,indent)
        else:
            inputs=n.inputs[:1] if op in {'reshape','cast_like','cast_to','to_fp32'} else n.inputs
            for value in inputs:
                inverse=singleton_reshape(graph.values[value].shape,graph.values[v].shape,known) if known is not None else None
                remember(value,at,inverse)
            args=[get(x,at,mask,indent) for x in inputs]
            if op in POINTWISE_EXPRESSION:body=_pointwise_expression(n,args,graph.values[v].dtype)
            elif op in {'reshape','cast_like','cast_to','to_fp32'}:body=args[0]
            elif op=='where':body=f'tl.where({args[0]}, {args[1]}, {args[2]})'
            elif op=='isfinite':body=f'({args[0]} == {args[0]}) & (tl.abs({args[0]}) != float("inf"))'
            elif op=='div':body=f'{args[0]}.to(tl.float32) / {args[1]}.to(tl.float32)'
            elif op in VALUE_LAYER:
                from compiler.backend.value_layer import triton_expression
                body=triton_expression(op,tuple(args),sigmoid_impl=config.sigmoid_impl)
            else:body=f'tl.{op}({args[0]}.to(tl.float32))'
        return emit(f'({body}).to({dtype})',identifier(v)+'_value',indent)
    def row_element(v,at=row,mask=rv,indent='    '):
        if v not in row_cache:
            n=produced.get(v)
            assert n is None or n.op not in REDUCTIONS, f'row statistic not ready: {v}'
            row_cache[v]=compute(v,at,mask,indent,row_element)
        return row_cache[v]
    def base_element(v,at,mask,indent='        '):
        if v in base_values and row_key(graph.values[v].shape)==base_row:
            return row_element(v)
        key=('base',v,at,mask)
        if tuple(graph.values[v].shape)==tuple(base):coordinates[v,at]=base_coordinates
        if key not in cache:cache[key]=compute(v,at,mask,indent,base_element)
        return cache[key]
    def wide_element(v,at,mask,indent='        '):
        if v in row_cache:return row_cache[v]
        if v in base_values:
            if row_key(graph.values[v].shape)==base_row:return row_element(v)
            return f'{base_element(v,base_at,base_mask)}[:, None]'
        key=('wide',v,at,mask)
        if key not in cache:cache[key]=compute(v,at,mask,indent,wide_element)
        return cache[key]
    if segmentation is not None:
        segment=emit('tl.program_id(1).to(tl.int64)' if stage=='max'
                     else 'tl.program_id(2).to(tl.int64)' if stage in {'partials','local'} else '0','segment')
        origin=f'{segment} * {rt} + '
        if stage in {'partials','finish','merge'}:
            part=emit(f'tl.arange(0, {st})','part')
            valid=emit(f'{rv} & ({part} < {sp})','part_valid')
            maxima=emit(f'tl.load({workptrs[0]} + {row} * {sp} + {part}, mask={valid}, other=float("-inf"))','maxima')
            maximum=emit(f'tl.where(tl.sum(({maxima} != {maxima}).to(tl.int32), 0) > 0, float("nan"), tl.max({maxima}, 0))','global_max')
            row_cache[base_reductions[0].output]=maximum
        if stage in {'finish','merge'}:
            sums=emit(f'tl.load({workptrs[1]} + {row} * {sp} + {part}, mask={valid}, other=0)','denominator_parts')
            if stage=='finish':row_cache[base_reductions[1].output]=emit(f'tl.sum({sums}, 0)','denominator')
            nums=emit(f'tl.load({workptrs[2]} + ({row} * {sp} + {part}[:, None]) * {expanded} + {d}[None, :], mask={valid}[:, None] & {dv}[None, :], other=0)','numerator_parts')
            if stage=='finish':row_cache[wide_reduction.output]=emit(f'tl.sum({nums}, 0)','numerator')
            else:
                factor=emit(f'tl.where({valid}, tl.{segmentation.merge.exp_op}({maxima} - {maximum}), 0.0)','correction')
                den=emit(f'tl.sum({sums} * {factor}, 0)','merged_denominator')
                num=emit(f'tl.sum({nums} * {factor}[:, None], 0)','merged_numerator')
                risk=emit(f'tl.load({workptrs[3]} + ({row} * {sp} + {part}[:, None]) * {expanded} + {d}[None, :], mask={valid}[:, None] & {dv}[None, :], other=0)','risk_parts')
                fallback=emit(f'(tl.sum(tl.sum({risk}, 0), 0) > 0) | ({den} != {den}) | (tl.abs({den}) == float("inf")) | (tl.sum((({num} != {num}) | (tl.abs({num}) == float("inf"))).to(tl.int32), 0) > 0)','needs_replay')
                row_cache[base_reductions[1].output]=den
                row_cache[wide_reduction.output]=num
    def reduce_sweeps():
        nonlocal base_at,base_mask,base_coordinates
        for n in group.members:
            if n is wide_reduction:break
            if segmentation is not None and (stage=='finish' or n.output in row_cache):continue
            if n.output not in base_values:continue
            if n.op not in REDUCTIONS:
                if row_key(graph.values[n.output].shape)==base_row:row_element(n.output)
                continue
            init='0.0' if n.op=='sum_last' else "float('-inf')"
            acc=emit(f'tl.full((), {init}, tl.float32)','base_reduced')
            if n is shared:
                row_cache[n.output]=acc
                continue
            r,step=names.fresh('r'),names.fresh('step')
            lines.extend([f'    {r} = tl.arange(0, {rt})',f'    for {step} in range({chunks}):'])
            base_at=emit(f'{row} * {width} + {origin}{step} * {rt} + {r}','base_index','        ')
            base_coordinates=tuple(row_coordinates)+(f'{origin}{step} * {rt} + {r}',)
            base_mask=emit(f'{rv} & ({origin}{step} * {rt} + {r} < {width})','base_valid','        ')
            value=base_element(n.inputs[0],base_at,base_mask)
            value=emit(f'tl.where({base_mask}, {value}.to(tl.float32), {init})','base_tile','        ')
            if stage=='local' and n is base_reductions[0]:
                lines.append(f'        {base_risk} = {base_risk} | (tl.sum(({base_mask} & (({value} != {value}) | (tl.abs({value}) == float("inf")))).to(tl.int32), 0) > 0)')
            if n.op=='sum_last':lines.append(f'        {acc} = {acc} + tl.sum({value}, 0)')
            else:
                mx=emit(f'tl.max({value}, 0)','base_max','        ')
                lines.append(f'        {acc} = tl.where(({acc} != {acc}) | (tl.sum(({value} != {value}).to(tl.int32), 0) > 0), float("nan"), tl.maximum({acc}, {mx}))')
            row_cache[n.output]=acc
            if segmentation is not None and stage=='max':
                lines.append(f'    tl.store({workptrs[0]} + {row} * {sp} + {segment}, {acc}, mask={rv})')
                return _make_kernel(name,'primitive',group,pairs,lines,f'({count}, triton.cdiv({width}, {rexpr}))',graph,
                    launch_options=(('num_warps',config.num_warps),('enable_fp_fusion',False)),
                    index_dtype=index_dtype,index_bound=index_proof)
        # Finish consumes all partials; no original R scan or input payload load.
        if segmentation is None or stage!='finish':
            r,step=names.fresh('r'),names.fresh('step');acc=emit(f'tl.full(({dt},), 0.0, tl.float32)','wide_reduced')
            lines.extend([f'    {r} = tl.arange(0, {rt})',f'    for {step} in range({chunks}):'])
            base_at=emit(f'{row} * {width} + {origin}{step} * {rt} + {r}','base_index','        ')
            base_coordinates=tuple(row_coordinates)+(f'{origin}{step} * {rt} + {r}',)
            base_mask=emit(f'{rv} & ({origin}{step} * {rt} + {r} < {width})','base_valid','        ')
            at=emit(f'({row} * {expanded} + {d}[None, :]) * {width} + ({origin}{step} * {rt} + {r})[:, None]','wide_index','        ')
            mask=emit(f'{base_mask}[:, None] & {dv}[None, :]','wide_valid','        ')
            coordinates[wide_reduction.inputs[0],at]=tuple(row_coordinates)+(f'{d}[None, :]',f'({origin}{step} * {rt} + {r})[:, None]')
            if shared is not None:
                value=base_element(shared.inputs[0],base_at,base_mask)
                value=emit(f'tl.where({base_mask}, {value}.to(tl.float32), 0.0)','base_tile','        ')
                if stage in {'partials','local'}:
                    lines.append('        if tl.program_id(1) == 0:')
                    lines.append(f'            {row_cache[shared.output]} = {row_cache[shared.output]} + tl.sum({value}, 0)')
                else:
                    lines.append(f'        {row_cache[shared.output]} = {row_cache[shared.output]} + tl.sum({value}, 0)')
            value=wide_element(wide_reduction.inputs[0],at,mask)
            value=emit(f'tl.where({mask}, {value}.to(tl.float32), 0.0)','wide_tile','        ')
            if stage=='local':
                lines.append(f'        {wide_risk} = {wide_risk} | (tl.sum((({value} != {value}) | (tl.abs({value}) == float("inf"))).to(tl.int32), 0) > 0)')
            lines.append(f'        {acc} = {acc} + tl.sum({value}, 0)');row_cache[wide_reduction.output]=acc
    if stage=='local':
        base_risk=emit('tl.full((), 0, tl.int32)','base_risk')
        wide_risk=emit(f'tl.full(({dt},), 0, tl.int32)','wide_risk')
    if stage not in {'finish','merge'}:
        kernel=reduce_sweeps()
        if kernel is not None:return kernel
    if stage=='merge':
        # Replay the original global-shift expression only for exceptional
        # tiles. The four scratch values remain visible in the stage IR.
        ready={base_reductions[0].output:maximum,
               base_reductions[1].output:den,wide_reduction.output:num}
        row_cache.clear();row_cache[base_reductions[0].output]=maximum;cache.clear()
        origin='';chunks=f'tl.cdiv({width}, {rt})'
        begin=len(lines);reduce_sweeps()
        lines.append(f'    {den} = {row_cache[base_reductions[1].output]}')
        lines.append(f'    {num} = {row_cache[wide_reduction.output]}')
        replay=lines[begin:];del lines[begin:]
        lines.append(f'    if {fallback}:');lines.extend('    '+line for line in replay)
        row_cache.clear();row_cache.update(ready);cache.clear()
    if stage=='local':
        wide=row_cache[wide_reduction.output]
        risk=emit(f'{wide_risk} | {base_risk} | ({wide} != {wide}) | (tl.abs({wide}) == float("inf"))','local_risk')
        lines.append(f'    tl.store({workptrs[2]} + ({row} * {sp} + {segment}) * {expanded} + {d}, {wide}, mask={rv} & {dv})')
        lines.append(f'    tl.store({workptrs[3]} + ({row} * {sp} + {segment}) * {expanded} + {d}, {risk}.to(tl.float32), mask={rv} & {dv})')
        lines.append('    if tl.program_id(1) == 0:')
        lines.append(f'        tl.store({workptrs[0]} + {row} * {sp} + {segment}, {row_cache[base_reductions[0].output]}, mask={rv})')
        lines.append(f'        tl.store({workptrs[1]} + {row} * {sp} + {segment}, {row_cache[base_reductions[1].output]}, mask={rv})')
        return _make_kernel(name,'primitive',group,pairs,lines,
            f'({count}, triton.cdiv({expanded}, {dexpr}), triton.cdiv({width}, {rexpr}))',graph,
            launch_options=(('num_warps',config.num_warps),('enable_fp_fusion',False)),
            index_dtype=index_dtype,index_bound=index_proof)
    if segmentation is not None and stage=='partials':
        lines.append(f'    tl.store({workptrs[2]} + ({row} * {sp} + {segment}) * {expanded} + {d}, {row_cache[wide_reduction.output]}, mask={rv} & {dv})')
        lines.append('    if tl.program_id(1) == 0:')
        lines.append(f'        tl.store({workptrs[1]} + {row} * {sp} + {segment}, {row_cache[base_reductions[1].output]}, mask={rv})')
        return _make_kernel(name,'primitive',group,pairs,lines,
            f'({count}, triton.cdiv({expanded}, {dexpr}), triton.cdiv({width}, {rexpr}))',graph,
            launch_options=(('num_warps',config.num_warps),('enable_fp_fusion',False)),
            index_dtype=index_dtype,index_bound=index_proof)
    # Outputs from base-only ancestry have one writer, on expanded tile 0.
    def depends_wide(v):
        if v==wide_reduction.output:return True
        n=produced.get(v)
        if not n:return False
        inputs=n.inputs[:1] if n.op in {'cast_like','cast_to','broadcast_like','reshape','to_fp32','move_first_last'} else n.inputs
        return any(depends_wide(x) for x in inputs)
    for v in results:
        if depends_wide(v):
            logical=emit(f'{row} * {expanded} + {d}','output_index')
            output=wide_element(v,logical,f'{rv} & {dv}',indent='    ')
            lines.append(f'    tl.store({ptrs[v]} + {logical}, {output}, mask={rv} & {dv})')
        else:
            output=row_element(v)
            lines.append(f'    if tl.program_id(1) == 0:')
            lines.append(f'        tl.store({ptrs[v]} + {row}, {output}, mask={rv})')
    return _make_kernel(name,'primitive',group,pairs,lines,
        f'({count}, triton.cdiv({expanded}, {dexpr}))',graph,
        launch_options=(('num_warps',config.num_warps),('enable_fp_fusion',False)),
        index_dtype=index_dtype,index_bound=index_proof)
