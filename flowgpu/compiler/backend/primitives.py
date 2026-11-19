"""Standalone compositional v0.3 lowering and indexed-region dispatch.

Unfused primitives remain available for source-identical performance controls.
"""

from flowgpu.compiler.shared import dimension_symbols
from flowgpu.compiler.backend.value_layer import NAMES as VALUE_LAYER


PRIMITIVES = ('exp', 'exp2', 'log', 'log2', 'to_fp32', 'isfinite', 'where',
              'zeros_like', 'ones_like', 'neg_inf_like', 'broadcast_like',
              'cast_like', 'cast_to', 'move_first_last', 'gather_row_offset', 'gather_rows', 'div',
              'sum_last', 'max_last') + VALUE_LAYER


def tile_offset(offset, index):
    """An address expression that is a *tile*, even when the source is a scalar.

    `broadcast_like` of a `[1,1]` source maps every lane to offset 0, and a bare
    `tl.load(ptr + 0, mask=<tile>)` is rejected by Triton: the pointer is a
    scalar and the mask is a block. Found on the H100 (2026-10-20) by the MoE
    router example, whose softcapping constant is a `[1,1]` input. Adding
    `index * 0` makes the address a tile of zeros and leaves it unchanged.
    """
    return offset if offset != '0' else f'({index} * 0)'


def offset_for(shape, target, index):
    from flowgpu.compiler.backend.backends import product
    padded = ['1'] * (len(target) - len(shape)) + list(shape)
    terms = []
    for axis, dimension in enumerate(padded):
        if dimension == '1':
            continue
        terms.append(f'(({index}) // ({product(target[axis+1:])}) % ({target[axis]})) * ({product(padded[axis+1:])})')
    return ' + '.join(terms) or '0'


def triton_kernel(graph, group, memory, name, config, *, autotune_fold=False):
    if group.row_gather or (len(group.members) == 1 and group.anchor.op == 'gather_rows'):
        from flowgpu.compiler.backend.row_gather_backend import triton_kernel as gather_kernel
        return gather_kernel(graph, group, memory, name, config)
    if group.indexed:
        from flowgpu.compiler.backend.indexed_backend import triton_kernel as indexed_kernel
        return indexed_kernel(graph, group, memory, name, config, autotune_fold=autotune_fold)
    from flowgpu.compiler.backend.backends import (Argument, Names, identifier, product,
                                           group_operands, group_results,
                                           scalar_operands, tensor_operands,
                                           unique_formal)
    from flowgpu.compiler.backend.triton_backend import (_input_layouts, _view_offset, _make_kernel,
                                                 grid_limited_body, row_limited_body, DTYPE_TO_TRITON)
    node = group.anchor
    operands, results = group_operands(graph, group), group_results(graph, group)
    result = node.output
    shape = graph.values[result].shape
    count = product(shape)
    names = Names(graph)
    # A scalar operand is a host number: one plain formal, no pointer and no
    # layout metadata. Its "load" is the formal itself.
    scalars = scalar_operands(graph, operands)
    tensors = tensor_operands(graph, operands)
    ptrs = {v: names.fresh(identifier(v)+'_ptr') for v in tensors + results}
    pairs = [(ptrs[v], Argument('tensor', v)) for v in tensors + results]
    block = names.fresh('FG_BLOCK')
    used_formals = {block} | dimension_symbols(graph)
    used_formals.update(parameter for parameter, _ in pairs)
    scalar_formals = {}
    for v in scalars:
        formal = unique_formal(identifier(v), used_formals)
        names.reserve(formal)
        scalar_formals[v] = formal
        pairs.append((formal, Argument('scalar', v)))
    for tensor in [graph.values[v] for v in tensors + results]:
        for d in tensor.shape:
            if not str(d).isdigit() and not any(p == f'{d}: tl.constexpr' for p, _ in pairs):
                pairs.append((f'{d}: tl.constexpr', Argument('value', str(d))))
    layouts = _input_layouts(graph, memory, tensors, names, pairs)
    pairs.append((f'{block}: tl.constexpr', Argument('value', str(config.default_block_size))))
    idx = names.fresh('idx'); mask = names.fresh('valid'); answer = names.fresh('answer')
    lines = [f'    {idx} = tl.program_id(0) * {block} + tl.arange(0, {block})',
             f'    {mask} = {idx} < ({count})']
    def load(v, offset=None, valid=None, other='0'):
        if v in scalar_formals:
            # A rank-0 operand is a host number.  It is wrapped as a 0-d tile so
            # the shared operator templates, which call `.to(...)` on their
            # operands, work on it exactly as they do on a loaded tile.
            return (f"tl.full((), {scalar_formals[v]}, "
                    f"{DTYPE_TO_TRITON[graph.values[v].dtype]})")
        at = _view_offset(v, idx if offset is None else offset, layouts, names, lines,
                          linearize=config.linear_view_indexing)
        var = names.fresh(identifier(v)+'_value')
        lines.append(f'    {var} = tl.load({ptrs[v]} + {at}, mask={mask if valid is None else valid}, other={other})')
        return var
    op = node.op
    x = node.inputs[0]
    if op in {'sum_last', 'max_last'}:
        # One row per program, streaming bounded chunks: no unbounded tile.
        width = graph.values[x].shape[-1]
        limit = config.reduction_block_limit or 128
        reduction_block = min(limit, (1 << (int(width)-1).bit_length()) if str(width).isdigit() else limit)
        if autotune_fold:
            reduction_block = names.fresh('FG_REDUCE_TILE')
            pairs.append((f'{reduction_block}: tl.constexpr',Argument('value',str(limit))))
        row = names.fresh('row'); col = names.fresh('col'); step = names.fresh('step'); value = names.fresh('tile')
        init = '0.0' if op == 'sum_last' else "float('-inf')"
        nan_seen = names.fresh('nan_seen')
        nan_count = names.fresh('nan_count')
        lines = [f'    {row} = tl.program_id(0)',
                 f'    {col} = tl.arange(0, {reduction_block})',
                 f'    {answer} = tl.full((), {init}, tl.float32)']
        if op == 'max_last':
            # NaN must propagate: `tl.max` follows the hardware reduction and
            # drops it, while the reference (and the fused reduction lowering)
            # return NaN for a row containing one (review P2, 2026-10-26).
            lines.append(f'    {nan_seen} = tl.full((), 0.0, tl.float32)')
        lines += [f'    for {step} in range(tl.cdiv({width}, {reduction_block})):',
                 f'        {idx} = {row} * {width} + {step} * {reduction_block} + {col}']
        address_lines=[]
        at=_view_offset(x, idx, layouts, names, address_lines)
        lines += ['    '+line for line in address_lines]
        lines += [f'        {value} = tl.load({ptrs[x]} + {at}, mask={step} * {reduction_block} + {col} < {width}, other={init}).to(tl.float32)']
        expr = f'{answer} + tl.sum({value}, 0)' if op == 'sum_last' else f'tl.maximum({answer}, tl.max({value}, 0))'
        if op == 'max_last':
            lines += [
                f'        {nan_count} = tl.sum(({value} != {value}).to(tl.float32), 0)',
                f'        {nan_seen} = {nan_seen} + {nan_count}',
            ]
        lines += [f'        {answer} = {expr}']
        if op == 'max_last':
            lines.append(f'    {answer} = tl.where({nan_seen} > 0, float("nan"), {answer})')
        lines.append(f'    tl.store({ptrs[result]} + {row}, {answer})')
        grid=f'({count},)'
        rows_are_programs = True
    else:
        if op in {'ones_like', 'zeros_like', 'neg_inf_like'}:
            expr = {'ones_like':'1', 'zeros_like':'0', 'neg_inf_like':"float('-inf')"}[op]
        elif op == 'move_first_last':
            old = graph.values[x].shape
            logical = f'({idx} % {old[0]}) * ({product(old[1:])}) + {idx} // {old[0]}'
            expr=load(x,logical)
        elif op == 'broadcast_like':
            expr=load(x,tile_offset(offset_for(graph.values[x].shape, shape, idx), idx))
        elif op == 'gather_row_offset':
            index=load(node.inputs[1]); logical=names.fresh('logical'); total=product(graph.values[x].shape)
            lines += [f'    {logical} = {idx}.to(tl.int64) * {graph.values[x].shape[1]} + {index}.to(tl.int64)',
                      f'    {logical} = tl.where({logical} < 0, {logical} + ({total}), {logical})']
            expr=load(x, logical, f'{mask} & ({logical} >= 0) & ({logical} < ({total}))')
        else:
            a=load(x)
            if op == 'where':
                b=load(node.inputs[1]);c=load(node.inputs[2]);expr=f'tl.where({a}, {b}, {c})'
            elif op == 'div':
                b=load(node.inputs[1]);expr=f'{a}.to(tl.float32) / {b}.to(tl.float32)'
            elif op in {'cast_like', 'cast_to', 'to_fp32'}:
                expr=a
            elif op == 'isfinite':
                expr=f'({a} == {a}) & (tl.abs({a}) != float("inf"))'
            elif op in VALUE_LAYER:
                # One row per operator in value_layer.TRITON, so the four sites
                # that emit an elementwise expression cannot disagree about
                # `tl.math.tanh` being different from `tl.tanh`. The one row a
                # platform may re-spell (sigmoid) resolves through the same
                # `PlatformConfig` policy the rest of the lowering reads.
                from flowgpu.compiler.backend.value_layer import triton_expression
                expr=triton_expression(op, (a,) + tuple(load(operand) for operand in node.inputs[1:]),
                                       sigmoid_impl=config.sigmoid_impl)
            else:
                expr=f'tl.{op}({a}.to(tl.float32))'
        value_dtype = graph.values[result].dtype
        if value_dtype == 'bool':
            # A comparison already yields i1; `.to(tl.int1)` is redundant on CUDA
            # and an explicit i1 cast is what the Ascend compiler may reject.
            # The STORE is not optional: without it the bool output is the
            # uninitialized `torch.empty` slot (review P0-2, 2026-10-26).
            lines += [f'    {answer} = {expr}',
                      f'    tl.store({ptrs[result]} + {idx}, {answer}, mask={mask})']
        else:
            lines += [f'    {answer} = ({expr}).to({DTYPE_TO_TRITON[value_dtype]})' if expr not in {'0','1',"float('-inf')"}
                      else f'    {answer} = tl.full(({block},), {expr}, {DTYPE_TO_TRITON[value_dtype]})',
                  f'    tl.store({ptrs[result]} + {idx}, {answer}, mask={mask})']
        grid=f'(triton.cdiv({count}, {config.default_block_size}),)'
        rows_are_programs = False
        # The same launch cap as the fused pointwise emitter: a target that
        # refuses a grid beyond its hardware extent must not be handed one.
        lines, grid = grid_limited_body(lines, grid, config, count=count, numel=count,
                                        block=block, grid_block=str(config.default_block_size),
                                        tile=names.fresh('tile'))
    if rows_are_programs:
        # One row per program is the only grid with no tile size to shrink, so it
        # is capped by striding rows instead. Ascend rejects coreDim > 65535.
        lines, grid = row_limited_body(lines, grid, config, count=count, row=row)
    kernel = _make_kernel(name,'primitive',group,pairs,lines,grid,graph,
                         launch_options=(('num_warps',config.num_warps),('enable_fp_fusion',False)))
    if autotune_fold and op in {'sum_last','max_last'}:
        from flowgpu.compiler.backend.autotune import decorate_fold
        kernel = decorate_fold(kernel,config,'rowwise',(reduction_block,),(str(width),count))
    return kernel


def reference_kernel(graph, group, memory, name):
    if group.row_gather or (len(group.members) == 1 and group.anchor.op == 'gather_rows'):
        from flowgpu.compiler.backend.row_gather_backend import reference_kernel as gather_kernel
        return gather_kernel(graph, group, memory, name)
    if group.indexed:
        from flowgpu.compiler.backend.indexed_backend import reference_kernel as indexed_kernel
        return indexed_kernel(graph, group, memory, name)
    from flowgpu.compiler.backend.backends import (Argument, Names, identifier, product,
                                           group_operands, group_results,
                                           scalar_operands, tensor_operands,
                                           unique_formal)
    from flowgpu.compiler.backend.reference import _kernel, _dimension_parameters, _rounded
    node=group.anchor;operands=group_operands(graph,group);results=group_results(graph,group)
    names=Names(graph)
    scalars=scalar_operands(graph,operands);tensors=tensor_operands(graph,operands)
    variables={v:names.fresh(identifier(v)+'_data') for v in tensors+results}
    pairs=[(variables[v],Argument('tensor',v)) for v in tensors+results]
    used_formals={parameter for parameter,_ in pairs} | dimension_symbols(graph)
    scalar_formals={}
    for v in scalars:
        formal=unique_formal(identifier(v),used_formals)
        names.reserve(formal)
        scalar_formals[v]=formal
        pairs.append((formal,Argument('scalar',v)))
    _dimension_parameters(names,pairs,[d for v in tensors+results for d in graph.values[v].shape])
    i=names.fresh('i');x=node.inputs[0];shape=graph.values[node.output].shape
    lines=[f'    for {i} in range({product(shape)}):']
    def at(v, offset=i):
        if v in scalar_formals:
            return scalar_formals[v]
        return f'{variables[v]}[{offset}]'
    a=at(x);op=node.op
    if op in {'sum_last','max_last'}:
        width=graph.values[x].shape[-1];j=names.fresh('j')
        row=f'{variables[x]}[{i}*{width}:({i}+1)*{width}]'
        expr=f'sum({row})' if op=='sum_last' else f'(math.nan if any(math.isnan(v) for v in {row}) else max({row}))'
    elif op=='move_first_last':
        old=graph.values[x].shape;expr=at(x,f'({i} % {old[0]}) * ({product(old[1:])}) + {i} // {old[0]}')
    elif op=='broadcast_like':expr=at(x,tile_offset(offset_for(graph.values[x].shape,shape,i), i))
    elif op=='gather_row_offset':
        logical=names.fresh('logical');total=product(graph.values[x].shape)
        lines += [f'        {logical} = {i} * {graph.values[x].shape[1]} + {at(node.inputs[1])}',
                  f'        if {logical} < 0: {logical} += {total}']
        expr=f'({at(x,logical)} if 0 <= {logical} < {total} else 0)'
    elif op=='where':expr=f'({at(node.inputs[1])} if {a} else {at(node.inputs[2])})'
    elif op=='isfinite':expr=f'math.isfinite({a})'
    elif op in {'cast_like', 'cast_to','to_fp32'}:expr=a
    elif op in {'ones_like','zeros_like','neg_inf_like'}:expr={'ones_like':'1','zeros_like':'0','neg_inf_like':'-math.inf'}[op]
    elif op=='div':expr=f'ieee_div({a}, {at(node.inputs[1])})'
    elif op in VALUE_LAYER:
        from flowgpu.compiler.backend.value_layer import reference_expression
        expr=reference_expression(op, tuple(at(operand) for operand in node.inputs))
    elif op in {'log','log2'}:expr=f'(math.{op}({a}) if {a} > 0 else (-math.inf if {a} == 0 else math.nan))'
    else:expr=f'safe_exp({a}, base2={op == "exp2"})'
    lines.append(f'        {variables[node.output]}[{i}] = '+_rounded(graph,node.output,expr))
    return _kernel(name,'primitive',group,operands,results,pairs,lines)
