"""Declared indexed write: `out = scatter_by_index(base, index, values)`.

This is the operator the catalogue survey put first (`research/language_gap_survey/`:
`indexed_write` blocked 23 tasks). It is *declared*, not general indexing -- the
index is an explicit tensor input, the ranks and the row width are static, and the
address is the index value rather than an expression the language computes.

Three semantics decisions are made here, not discovered later:

1. **Out-of-bounds entries are discarded.** That mirrors `gather_rows` returning
   zero for an out-of-bounds read. Rejecting would need a compile-time bound proof
   a data-dependent index cannot supply; clamping would silently duplicate a write
   onto a neighbouring row.
2. **Duplicate indices are last-write-wins, unordered.** Two values targeting one
   row race, so a program that depends on the order is not expressible this way.
3. **Two launches, not one.** A pure operator cannot rewrite its base, so the
   lowering is "copy the base, then write the addressed rows"; the copy must
   complete before the scattered writes. The emitter protocol allows a lowering to
   return a tuple of kernels for exactly this reason.

The access shape is `SCATTER`, so `analysis/access_shape.platform_conflicts`
refuses the plan on a target that cannot sustain an indexed write, and
`analysis/dependences` treats the resulting WAR/WAW as `may-overlap` -- which is
the cost the dependence guard on slot reuse exists to respect.
"""

from flowgpu.compiler.backend.backends import (
    Argument,
    Names,
    group_operands,
    group_results,
    identifier,
)


def _literal(graph, value, axis, what):
    extent = str(graph.values[value].shape[axis])
    if not extent.isdigit():
        raise ValueError(f"scatter_by_index requires a literal {what} extent")
    return int(extent)


def triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import TritonKernel

    assert len(group.members) == 1
    assert all(node.op == "scatter_by_index" for node in group.members)
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 3 and len(results) == 1
    base, index, values = operands
    out = results[0]
    rows = _literal(graph, base, 0, "destination row")
    writes = _literal(graph, values, 0, "value row")
    row_shape = [int(d) for d in graph.values[base].shape[1:]]
    width = 1
    for extent in row_shape:
        width *= extent
    if _literal(graph, values, 0, "value row") != writes:
        raise AssertionError("scatter_by_index row count disagrees with indptr")
    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr") for value in operands + results}
    pairs = [(ptr[value], Argument("tensor", value)) for value in operands + results]

    def constexpr(hint, value):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(value))))
        return symbol

    rows_symbol = constexpr("ROWS", rows)
    width_symbol = constexpr("WIDTH", width)
    block = names.fresh("BLOCK")
    block_size = config.default_block_size
    pairs.append((f"{block}: tl.constexpr", Argument("value", str(block_size))))

    # launch 1: the base is copied through, so untouched rows keep their values
    total = rows * width
    chunks = (total + block_size - 1) // block_size
    e = names.fresh("e")
    copy_lines = [
        f"    {e} = tl.program_id(0) * {block} + tl.arange(0, {block})",
        f"    keep = {e} < {rows_symbol} * {width_symbol}",
        f"    value = tl.load({ptr[base]} + {e}, mask=keep, other=0)",
        f"    tl.store({ptr[out]} + {e}, value, mask=keep)",
    ]
    copy_parameters = tuple(parameter for parameter, _ in pairs)
    copy_source = "# scatter_by_index (copy)\n@triton.jit\n"
    copy_source += f"def {name}_copy({', '.join(copy_parameters)}):\n" + "\n".join(copy_lines) + "\n"
    copy_kernel = TritonKernel(f"{name}_copy", "scatter_by_index",
                               tuple(node.op for node in group.members),
                               operands, results, copy_parameters,
                               tuple(argument for _, argument in pairs),
                               f"({chunks},)", copy_source,
                               launch_options=(("num_warps", config.num_warps),))

    # launch 2: each value row goes to its indexed destination row, if in bounds
    write_total = writes * width
    write_chunks = (write_total + block_size - 1) // block_size
    flat = names.fresh("flat")
    inside = names.fresh("inside")
    row = names.fresh("row")
    offset = names.fresh("offset")
    target = names.fresh("target")
    in_bounds = names.fresh("in_bounds")
    value = names.fresh("value")
    # A 64-bit index must stay 64-bit (review P1-7, 2026-10-26): a wrapped
    # int32 destination writes a legal row instead of being masked out.
    index_dtype = "tl.int64" if graph.values[index].dtype == "int64" else "tl.int32"
    write_lines = [
        f"    {flat} = tl.program_id(0) * {block} + tl.arange(0, {block})",
        f"    {inside} = {flat} < {writes} * {width_symbol}",
        f"    {row} = {flat} // {width_symbol}",
        f"    {offset} = {flat} - {row} * {width_symbol}",
        f"    {target} = tl.load({ptr[index]} + {row}, mask={inside}, other=-1).to({index_dtype})",
        f"    {in_bounds} = {inside} & ({target} >= 0) & ({target} < {rows_symbol})",
        f"    {value} = tl.load({ptr[values]} + {flat}, mask={inside}, other=0)",
        f"    tl.store({ptr[out]} + {target} * {width_symbol} + {offset}, {value},"
        f" mask={in_bounds})",
    ]
    write_parameters = tuple(parameter for parameter, _ in pairs)
    write_source = "# scatter_by_index (write)\n@triton.jit\n"
    write_source += f"def {name}_write({', '.join(write_parameters)}):\n" + "\n".join(write_lines) + "\n"
    write_kernel = TritonKernel(f"{name}_write", "scatter_by_index",
                                tuple(node.op for node in group.members),
                                operands, results, write_parameters,
                                tuple(argument for _, argument in pairs),
                                f"({write_chunks},)", write_source,
                                launch_options=(("num_warps", config.num_warps),))
    return (copy_kernel, write_kernel)


def reference_kernel(graph, group, memory, name):
    """Host-side reference: copy the base, then write each in-bounds row."""
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    base, index, values = operands
    out = results[0]
    rows = _literal(graph, base, 0, "destination row")
    writes = _literal(graph, values, 0, "value row")
    width = 1
    for extent in [int(d) for d in graph.values[base].shape[1:]]:
        width *= extent
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs,
                          [d for v in operands + results for d in graph.values[v].shape])
    lines = [
        f"    for flat in range({rows} * {width}):",
        f"        {variables[out]}[flat] = {variables[base]}[flat]",
        f"    for row in range({writes}):",
        f"        target = {variables[index]}[row]",
        f"        if 0 <= target < {rows}:",
        f"            for offset in range({width}):",
        f"                {variables[out]}[target * {width} + offset] = "
        f"{_rounded(graph, out, variables[values] + f'[row * {width} + offset]')}",
    ]
    return _kernel(name, "scatter_by_index", group, operands, results, pairs, lines)
