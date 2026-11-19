"""Emit sequential sweeps in a shared row/column domain with several outputs.

Row values are computed once per row tile, outside column loops. Full-domain
outputs can be stored during their consuming reduction sweep. A final sweep
handles any other full-domain outputs. Dtype boundaries survive every node.
"""
from flowgpu.compiler.shared import dimension_symbols
from flowgpu.compiler.backend.backends import Argument, Names, identifier, product, group_operands, group_results
from flowgpu.compiler.optimization.indexed_fusion import REDUCTIONS
from flowgpu.compiler.optimization.reduction_fusion import row_key
from flowgpu.compiler.backend.primitives import offset_for, tile_offset


def triton_kernel(graph, group, memory, name, config, *, autotune_fold=False):
    from flowgpu.compiler.backend.triton_backend import (_input_layouts, _tensor_layouts, _view_offset, _make_kernel,
                                        DTYPE_TO_TRITON, POINTWISE_EXPRESSION, _pointwise_expression,
                                        grid_limited_body, grid_ceiling, capped_grid)
    from flowgpu.compiler.backend.value_layer import NAMES as VALUE_LAYER
    operands, results = group_operands(graph,group), group_results(graph,group)
    produced = {n.output:n for n in group.members}
    domain = group.reduction_domain
    width, count = domain[-1],product(domain[:-1])
    row_shape = row_key(domain[:-1])
    is_row = lambda v:row_key(graph.values[v].shape)==row_shape
    # An explicitly single-row tile has no second vector dimension. Keeping
    # it rank one avoids a needless [columns,1] layout and permits backends
    # that cannot legalize axis-zero reductions on rank-two tiles.
    # Fold autotuning may replace FG_ROWS with a multirow candidate. Its
    # META grid and vector shape must continue to agree for every candidate.
    scalar_row = config.indexed_rows == 1 and not autotune_fold
    row_broadcast = lambda v: v if scalar_row else f'{v}[None, :]'
    names=Names(graph)
    # Rank-0 operands are host numbers. They become plain kernel arguments and
    # are referenced by name wherever a tile would be loaded: a reduction whose
    # epilogue divides by a scalar (dynamic_quant's scale, or a normalisation
    # dividing by the row extent) is otherwise inexpressible.
    from flowgpu.compiler.backend.triton_backend import scalar_operands, tensor_operands, unique_formal
    scalars=scalar_operands(graph,operands)
    tensors=tensor_operands(graph,operands)
    ptrs={v:names.fresh(identifier(v)+'_ptr') for v in tensors+results}
    pairs=[(ptrs[v],Argument('tensor',v)) for v in tensors+results]
    scalar_formals={}
    used_formals=({parameter.split(':')[0] for parameter,_ in pairs}
                  | dimension_symbols(graph))
    for value in scalars:
        formal=unique_formal(identifier(value),used_formals)
        names.reserve(formal);scalar_formals[value]=formal
        pairs.append((formal,Argument('scalar',value)))
    dimensions=[]
    for v in operands+tuple(produced):
        for d in graph.values[v].shape:
            if not str(d).isdigit() and d not in dimensions:
                dimensions.append(d);pairs.append((f'{d}: tl.constexpr',Argument('value',str(d))))
    layouts=_input_layouts(graph,memory,tensors,names,pairs)
    if config.strided_reduction_outputs:
        layouts.update(_tensor_layouts(graph,results,names,pairs))
    limit=config.reduction_block_limit or 128
    tile,rows=names.fresh('FG_COLUMNS'),names.fresh('FG_ROWS')
    column_limit=min(limit,config.indexed_columns or limit)
    tile_expr=f'min({column_limit}, triton.next_power_of_2({width}))'
    rows_expr=f'min({config.indexed_rows or 16}, max(1, {limit} // ({tile_expr})))'
    pairs.extend([(f'{tile}: tl.constexpr',Argument('value',tile_expr)),
                  (f'{rows}: tl.constexpr',Argument('value',rows_expr))])
    row,valid=names.fresh('row'),names.fresh('row_valid')
    row_expression = 'tl.program_id(0).to(tl.int64)' if scalar_row else f'tl.program_id(0).to(tl.int64) * {rows} + tl.arange(0, {rows})'
    lines=[f'    {row} = {row_expression}',
           f'    {valid} = {row} < ({count})']
    row_cache,full_cache={},{}
    reductions=[n for n in group.members if n.op in REDUCTIONS]
    published=set()

    def emit(expr,hint,indent='    '):
        var=names.fresh(hint);lines.append(f'{indent}{var} = {expr}');return var

    def store(v,at,value,mask,indent):
        address_lines=[];address=_view_offset(v,at,layouts,names,address_lines)
        lines.extend(indent+line[4:] for line in address_lines)
        lines.append(f'{indent}tl.store({ptrs[v]} + {address}, {value}, mask={mask})')

    def publish_row(v):
        if (config.eager_row_result_stores and v in results and is_row(v)
                and v in row_cache and v not in published):
            store(v,row,row_cache[v],valid,'    ')
            published.add(v)

    # A direct domain load already has row/column coordinates. Reconstructing
    # them from row*K+column repeats wide division by K inside every sweep.
    # Only exact-domain, unremapped inputs use this optimization; arbitrary
    # permutation/broadcast use sites retain the general address calculation.
    row_coordinates=[]
    if config.coalesce_reduction_sweeps:
        for axis,dimension in enumerate(domain[:-1]):
            tail=product(domain[axis+1:-1])
            coordinate=f'({row} // ({tail}))'
            if axis:coordinate+=f' % ({dimension})'
            row_coordinates.append(emit(coordinate,'row_coordinate'))
    direct_coordinates={}

    def load(v,at,mask,indent):
        coordinates=direct_coordinates.get(at) if tuple(graph.values[v].shape)==tuple(domain) else None
        address_lines=[];address=_view_offset(v,at,layouts,names,address_lines,coordinates=coordinates)
        lines.extend(indent+line[4:] for line in address_lines)
        return emit(f'tl.load({ptrs[v]} + {address}, mask={mask}, other=0)',identifier(v)+'_load',indent)

    def expression(node,args):
        op=node.op;dtype=graph.values[node.output].dtype
        if op in POINTWISE_EXPRESSION:return _pointwise_expression(node,args,dtype)
        if op in {'to_fp32','cast_like','cast_to','reshape'}:return args[0]
        if op=='where':return f'tl.where({args[0]}, {args[1]}, {args[2]})'
        if op=='isfinite':return f'({args[0]} == {args[0]}) & (tl.abs({args[0]}) != float("inf"))'
        if op=='div':return f'{args[0]}.to(tl.float32) / {args[1]}.to(tl.float32)'
        if op in VALUE_LAYER:
            from flowgpu.compiler.backend.value_layer import triton_expression
            return triton_expression(op,tuple(args),sigmoid_impl=config.sigmoid_impl)
        return f'tl.{op}({args[0]}.to(tl.float32))'

    def compute(v,at,mask,indent,get):
        if v in scalar_formals:
            # Reached from more than one materialisation path (`tail_value`
            # recomputes value ops from the accumulator tile), so the scalar
            # short-circuit lives here as well as in `full_element`.
            return scalar_tile(v)
        node=produced.get(v)
        if node is None:return load(v,at,mask,indent)
        op=node.op;x=node.inputs[0];dtype=DTYPE_TO_TRITON[graph.values[v].dtype]
        if op in {'ones_like','zeros_like','neg_inf_like'}:
            constant={'ones_like':'1','zeros_like':'0','neg_inf_like':"float('-inf')"}[op]
            body=f'tl.full((), {constant}, {dtype})'
        elif op=='move_first_last':
            old=graph.values[x].shape
            mapped=emit(f'(({at}) % {old[0]}) * ({product(old[1:])}) + ({at}) // {old[0]}','moved_index',indent)
            body=get(x,mapped,mask,indent)
        elif op=='broadcast_like':
            if is_row(x) and x in row_cache:
                body=row_cache[x] if indent=='    ' else row_broadcast(row_cache[x])
            else:
                mapped=emit(tile_offset(offset_for(graph.values[x].shape,graph.values[v].shape,at),at),'broadcast_index',indent)
                body=get(x,mapped,mask,indent)
        else:
            inputs=node.inputs[:1] if op in {'reshape','cast_like','cast_to','to_fp32'} else node.inputs
            body=expression(node,[get(x,at,mask,indent) for x in inputs])
        return emit(f'({body}).to({dtype})',identifier(v)+'_value',indent)

    def scalar_tile(v):
        """A rank-0 operand as a 0-d tile.

        The shared expression helpers call tensor methods (`.to(...)`) on their
        operands, which a bare Python scalar argument does not have.  A 0-d
        `tl.full` is the same number with the tile interface, and broadcasts
        against any tile shape.
        """
        dtype=DTYPE_TO_TRITON[graph.values[v].dtype]
        return f'tl.full((), {scalar_formals[v]}, {dtype})'

    def full_element(v,at,mask,indent='        '):
        if v in scalar_formals:
            # A host number needs no load and no address.
            return scalar_tile(v)
        if is_row(v):
            # Every row dependency has already been computed in topological
            # order before this sweep, including external row operands.
            assert v in row_cache, f'row dependency {v} was not ready'
            return row_broadcast(row_cache[v])
        key=v,at,mask,indent
        if key not in full_cache:full_cache[key]=compute(v,at,mask,indent,full_element)
        return full_cache[key]

    def row_element(v,at=row,mask=valid,indent='    '):
        if v not in row_cache:
            node=produced.get(v)
            assert node is None or node.op not in REDUCTIONS,'reduction must be scheduled before its row consumers'
            row_cache[v]=compute(v,at,mask,indent,row_element)
        return row_cache[v]

    def ancestors(v):
        found=set()
        def walk(x):
            if x in found:return
            found.add(x)
            node=produced.get(x)
            if node and node.op not in REDUCTIONS:
                inputs=node.inputs[:1] if node.op in {'reshape','cast_like','cast_to','to_fp32','broadcast_like','move_first_last'} else node.inputs
                if node.op not in {'ones_like','zeros_like','neg_inf_like'}:
                    for arg in inputs:walk(arg)
        walk(v);return found
    store_owner={}
    for reduction in reductions:
        for v in results:
            if not is_row(v) and v in ancestors(reduction.inputs[0]):store_owner[v]=reduction.output
    for v in operands:
        if is_row(v):row_element(v)

    def sweep(batch=(), outputs=()):
        col,step=names.fresh('col'),names.fresh('step')
        accumulators={}
        for reduction in batch:
            init='0.0' if reduction.op=='sum_last' else "float('-inf')"
            accumulator_shape = '()' if scalar_row else f'({rows},)'
            accumulators[reduction.output]=emit(f'tl.full({accumulator_shape}, {init}, tl.float32)','reduced')
        column_expression = f'tl.arange(0, {tile})' + ('' if scalar_row else '[:, None]')
        lines.extend([f'    {col} = {column_expression}',
                      f'    for {step} in range(tl.cdiv({width}, {tile})):'])

        def row_pure(value):
            """Whether a value cannot vary inside the column loop, i.e. per-row.

            The tail path stores a value once per row at the *row* address, so it
            is only legal for a value that is the same for every element of that
            row. Deciding that from "the chain produced a carrier address" is
            wrong: a chain that reaches a full-domain input also ends at a
            carrier (the row), and treating it as per-row hoists the computation
            out of the loop whose index it reads. The emitted kernel then names
            `full_index_2` outside the loop that binds it -- a device-side
            `NameError` -- and the elements it should have stored per column were
            never computed.
            """
            if value in accumulators or value in row_cache or value in scalar_formals:
                return True
            node = produced.get(value)
            if node is None:
                # A full-domain input: its elements differ inside the sweep.
                return False
            if node.op in REDUCTIONS:
                return True
            inputs = node.inputs[:1] \
                if node.op in {'reshape', 'cast_like', 'cast_to', 'to_fp32'} \
                else node.inputs
            return all(row_pure(argument) for argument in inputs)
        at=emit(f'{row_broadcast(row)} * {width} + {step} * {tile} + {col}','full_index','        ')
        if config.coalesce_reduction_sweeps:
            direct_coordinates[at]=tuple(row_broadcast(v) for v in row_coordinates)+(f'{step} * {tile} + {col}',)
        mask=emit(f'{row_broadcast(valid)} & ({step} * {tile} + {col} < {width})','full_valid','        ')
        # The expression cache shares loads and rounded subexpressions among
        # independent reductions in this loop. Each accumulator remains FP32.
        for reduction in batch:
            value=full_element(reduction.inputs[0],at,mask)
            init='0.0' if reduction.op=='sum_last' else "float('-inf')"
            value=emit(f'tl.where({mask}, {value}.to(tl.float32), {init})','reduce_tile','        ')
            acc=accumulators[reduction.output]
            if reduction.op=='sum_last':lines.append(f'        {acc} = {acc} + tl.sum({value}, 0)')
            else:
                maximum=emit(f'tl.max({value}, 0)','tile_max','        ')
                nan=f'({acc} != {acc}) | (tl.sum(({value} != {value}).to(tl.int32), 0) > 0)'
                lines.append(f'        {acc} = tl.where({nan}, float("nan"), tl.maximum({acc}, {maximum}))')
        # A result whose producer chain runs back to this batch's
        # accumulators is stored from registers ONCE, after the column
        # loop: the value ops (casts, same-shape pointwise) recompute on
        # the accumulator tile and the rotation links fold into the store
        # address's inverse map (13.6). Anything else is recomputed as a
        # gather inside the loop.
        def tail_value(value,indent='        '):
            """(value expression, address expression) for a tail result.

            The address starts at the per-row index and accumulates each
            rotation link's inverse map; value ops recompute on the
            accumulator tile. None means v is not a tail result.
            """
            node=produced.get(value)
            if value in accumulators:
                # The accumulator is per-ROW: its index is the row index,
                # not the domain's row*width position.
                return f'({accumulators[value]})',f'{row}'
            if value in row_cache and produced.get(value) is not None \
                    and produced[value].op in REDUCTIONS:
                return f'({row_cache[value]})',f'{row}'
            if node is None:
                return None
            if node.op=='move_first_last':
                inner=tail_value(node.inputs[0],indent)
                if inner is None:return None
                inner_value,inner_address=inner
                source=graph.values[node.inputs[0]].shape
                address=emit(f'(({inner_address}) % ({product(source[1:])})) * ({source[0]})'
                             f' + (({inner_address}) // ({product(source[1:])}))',
                             'store_rotation',indent)
                return inner_value,address
            if node.op in {'reshape','cast_like','cast_to','to_fp32'} or \
                    node.op in POINTWISE_EXPRESSION:
                # Casts read only their primary operand: the declared
                # dtype boundary is the whole computation, so the second
                # operand of cast_like is never gathered.
                needed = node.inputs[:1] \
                    if node.op in {'reshape','cast_like','cast_to','to_fp32'} \
                    else node.inputs
                parts=[]
                carrier=None
                for index,argument in enumerate(node.inputs):
                    if argument not in needed and index != 0:
                        continue
                    part=tail_value(argument,indent)
                    if part is None:
                        if is_row(argument) and argument in row_cache:
                            parts.append(f'({row_cache[argument]})')
                        else:
                            parts.append(full_element(argument,at,mask,indent))
                    else:
                        parts.append(part[0])
                        carrier=part[1]
                return expression(node,parts),(carrier or row)
            return None
        tail_outputs=[]
        for v in outputs:
            # A tail result is only legal for a per-row value: `row_pure` is what
            # decides that, not the mere existence of a carrier.
            tail=tail_value(v,'    ') if row_pure(v) else None
            if tail is not None:
                tail_outputs.append((v,tail))
            else:
                output=full_element(v,at,mask)
                store(v,at,output,mask,'        ')
        # Publish only after every source in the independent batch is emitted.
        row_cache.update(accumulators)
        for v in accumulators:publish_row(v)
        # Tail results are per-row: store them once the column loop closes,
        # at the inverse-mapped address, masked by row validity.
        for v,(value_expression,address) in tail_outputs:
            dtype=DTYPE_TO_TRITON[graph.values[v].dtype]
            output=emit(f'({value_expression}).to({dtype})',identifier(v)+'_value','    ')
            store(v,address,output,valid,'    ')

    batches=group.reduction_batches or tuple((n.output,) for n in reductions)
    by_output={n.output:n for n in reductions}
    starts={batch[0]:tuple(by_output[v] for v in batch) for batch in batches}
    for node in group.members:
        if node.op in REDUCTIONS:
            if node.output in starts:
                batch=starts[node.output]
                owners={r.output for r in batch}
                sweep(batch,[v for v,owner in store_owner.items() if owner in owners])
        elif is_row(node.output):
            row_element(node.output)
            publish_row(node.output)
    remaining=[v for v in results if not is_row(v) and v not in store_owner]
    if remaining:sweep(outputs=remaining)
    for v in results:
        if is_row(v) and v not in published:store(v,row,row_element(v),valid,'    ')
    options=[('num_warps',config.num_warps),('enable_fp_fusion',False)]
    # rows_expr collapses to 1 whenever the row is at least one reduction tile
    # wide, so this grid is the row count itself. Ascend rejects coreDim above
    # 65535, so the rows are strided instead of one program per row.
    lines, grid = grid_limited_body(lines, f'(triton.cdiv({count}, {rows_expr}),)', config,
                                    count=count, numel=count, block=rows, grid_block=rows_expr,
                                    tile=names.fresh('tile'))
    kernel = _make_kernel(name,'primitive',group,pairs,lines,grid,graph,
                         launch_options=tuple(options))
    if autotune_fold:
        from dataclasses import replace
        from flowgpu.compiler.backend.autotune import decorate_fold
        kernel = decorate_fold(kernel,config,'indexed_reduction',(tile,rows),(str(width),count))
        limit=grid_ceiling(config)
        folded=f"triton.cdiv({count}, META[{rows!r}])"
        kernel = replace(kernel,grid=f"lambda META: {capped_grid(folded, limit) if limit is not None else f'({folded},)'}")
    return kernel
