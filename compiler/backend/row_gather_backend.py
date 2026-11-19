"""General row selection and optional shared-index multi-output launches.

Row indices are normalized once per source height; invalid rows yield zero.
No host reads of index data or Torch arithmetic. Layout metadata handles all
source axes, including transposes, storage offsets and zero-stride views.
"""
from compiler.backend.backends import Argument, Names, identifier, product, group_operands, group_results


def triton_kernel(graph, group, memory, name, config):
    from compiler.optimization.row_gather_fusion import group_config
    config = group_config(group, config)
    from compiler.backend.triton_backend import _input_layouts, _view_offset, _make_kernel
    operands, results = group_operands(graph, group), group_results(graph, group)
    nodes = [n for n in group.members if n.output in results]
    assert nodes and all(n.op == 'gather_rows' for n in nodes)
    index = nodes[0].inputs[1]
    assert all(n.inputs[1] == index for n in nodes)
    names = Names(graph)
    ptrs = {v: names.fresh(identifier(v) + '_ptr') for v in operands + results}
    pairs = [(ptrs[v], Argument('tensor', v)) for v in operands + results]
    dims = []
    for v in operands + results:
        for d in graph.values[v].shape:
            if not str(d).isdigit() and d not in dims:
                dims.append(d)
                pairs.append((f'{d}: tl.constexpr', Argument('value', str(d))))
    layouts = _input_layouts(graph, memory, operands, names, pairs)
    widths = [product(graph.values[n.inputs[0]].shape[1:]) for n in nodes]
    largest = widths[0] if len(widths) == 1 else f'max({", ".join(widths)})'
    nrows = product(graph.values[index].shape)
    row, col = names.fresh('row'), names.fresh('column')
    lines = []
    flat_scan = None
    if config.gather_mapping == 'row_tiled':
        bm, bn = config.indexed_rows or 8, config.indexed_columns or 128
        if bm * bn > (config.reduction_block_limit or 4096):
            raise ValueError('row gather tile exceeds platform tile bound')
        lines.extend([f'    {row} = tl.program_id(0).to(tl.int64) * {bm} + tl.arange(0, {bm})',
                      f'    {col} = tl.program_id(1).to(tl.int64) * {bn} + tl.arange(0, {bn})'])
        grid = f'(triton.cdiv({nrows}, {bm}), triton.cdiv({largest}, {bn}))'
    else:
        block = config.default_block_size
        idx = names.fresh('offset')
        max_width = names.fresh('FG_MAX_WIDTH')
        pairs.append((f'{max_width}: tl.constexpr', Argument('value', largest)))
        lines.extend([f'    {idx} = tl.program_id(0).to(tl.int64) * {block} + tl.arange(0, {block})',
                      f'    {row} = {idx} // {max_width}', f'    {col} = {idx} % {max_width}'])
        grid = f'(triton.cdiv(({nrows}) * ({largest}), {block}),)'
        flat_scan = (idx, block)
    raw = names.fresh('raw_row')
    at = _view_offset(index, row, layouts, names, lines)
    lines.append(f'    {raw} = tl.load({ptrs[index]} + {at}, mask={row} < ({nrows}), other=0).to(tl.int64)')
    for node, width in zip(nodes, widths):
        src, out = node.inputs[0], node.output
        height = graph.values[src].shape[0]
        selected, logical, mask = names.fresh('selected_row'), names.fresh('logical'), names.fresh('valid')
        lines.append(f'    {selected} = tl.where({raw} < 0, {raw} + {height}, {raw})')
        rr, cc, ss = row, col, selected
        if config.gather_mapping == 'row_tiled':
            rr, cc, ss = row + '[:, None]', col + '[None, :]', selected + '[:, None]'
        output_mask = f'({rr} < ({nrows})) & ({cc} < ({width}))'
        lines.extend([f'    {mask} = ({output_mask}) & ({ss} >= 0) & ({ss} < {height})',
                      f'    {logical} = {ss} * ({width}) + {cc}'])
        address = _view_offset(src, logical, layouts, names, lines)
        value = names.fresh('gathered')
        lines.extend([f'    {value} = tl.load({ptrs[src]} + {address}, mask={mask}, other=0)',
                      f'    tl.store({ptrs[out]} + {rr} * ({width}) + {cc}, {value}, mask={output_mask})'])
    if flat_scan is not None and config.max_grid_dim is not None:
        # A flat gather scan is one tile per program, so Ascend's coreDim
        # ceiling applies exactly as it does to the pointwise emitters.
        from compiler.backend.triton_backend import grid_limited_body
        linear, block_size = flat_scan
        lines, grid = grid_limited_body(lines, grid, config,
                                        count=f'({nrows}) * ({largest})',
                                        numel=f'({nrows}) * ({largest})',
                                        block=str(block_size), grid_block=str(block_size),
                                        tile=names.fresh('tile'))
    return _make_kernel(name, 'primitive', group, pairs, lines, grid, graph,
                        launch_options=(('num_warps', config.num_warps), ('enable_fp_fusion', False)))


def reference_kernel(graph, group, memory, name):
    from compiler.backend.reference import _kernel, _dimension_parameters, _rounded
    operands, results = group_operands(graph, group), group_results(graph, group)
    names = Names(graph)
    variables = {v: names.fresh(identifier(v) + '_data') for v in operands + results}
    pairs = [(variables[v], Argument('tensor', v)) for v in operands + results]
    _dimension_parameters(names, pairs, [d for v in operands + results for d in graph.values[v].shape])
    lines = []
    for node in group.members:
        if node.output not in results:
            continue
        x, index = node.inputs
        width, height = product(graph.values[x].shape[1:]), graph.values[x].shape[0]
        i, selected = names.fresh('i'), names.fresh('selected_row')
        lines.extend([f'    for {i} in range({product(graph.values[node.output].shape)}):',
                      f'        {selected} = {variables[index]}[{i} // ({width})]',
                      f'        if {selected} < 0: {selected} += {height}'])
        expr = f'({variables[x]}[{selected} * ({width}) + {i} % ({width})] if 0 <= {selected} < {height} else 0)'
        lines.append(f'        {variables[node.output]}[{i}] = ' + _rounded(graph, node.output, expr))
    return _kernel(name, 'primitive', group, operands, results, pairs, lines)
