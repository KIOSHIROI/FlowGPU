"""`cache_write_hsd_range`: write a contiguous run of prompt rows."""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier)


def _literal(graph, value, axis, what):
    extent = str(graph.values[value].shape[axis])
    if not extent.isdigit():
        raise ValueError(f"{what} requires literal extents")
    return int(extent)


def cache_write_range_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import TritonKernel, _pointer
    assert len(group.members) == 1 and group.anchor.op == "cache_write_hsd_range"
    operands, results = group_operands(graph, group), group_results(graph, group)
    cache, new, loc = operands
    (out,) = results
    heads = _literal(graph, cache, 0, "cache_write_hsd_range")
    keys = _literal(graph, cache, 1, "cache_write_hsd_range")
    depth = _literal(graph, cache, 2, "cache_write_hsd_range")
    added = _literal(graph, new, 1, "cache_write_hsd_range")
    block = 1 << (depth - 1).bit_length()
    names = Names(graph)
    pairs = [(_pointer(value), Argument("tensor", value)) for value in (cache, new, loc, out)]
    pairs.append(("D: tl.constexpr", Argument("value", str(block))))
    row, d, h, s, start, index, old, value = (names.fresh(hint) for hint in
                                              ("row", "d", "h", "s", "start", "index", "old", "value"))
    source = f"""# cache_write_hsd_range: copy the cache, then overlay a prompt run
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
    {row} = tl.program_id(0)
    {d} = tl.arange(0, D)
    {h} = {row} // {keys}
    {s} = {row} % {keys}
    {start} = tl.load({_pointer(loc)})
    {index} = {s} - {start}
    {old} = tl.load({_pointer(cache)} + {row} * {depth} + {d}, mask={d} < {depth}, other=0.0)
    {value} = tl.load({_pointer(new)} + ({h} * {added} + {index}) * {depth} + {d},
                      mask=({index} >= 0) & ({index} < {added}) & ({d} < {depth}), other=0.0)
    {value} = tl.where(({index} >= 0) & ({index} < {added}), {value}, {old})
    tl.store({_pointer(out)} + {row} * {depth} + {d}, {value}, mask={d} < {depth})
"""
    return TritonKernel(name, "cache_write_hsd_range", ("cache_write_hsd_range",),
                        operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs), f"({heads * keys},)",
                        source, launch_options=(("num_warps", config.num_warps),))


def cache_write_range_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded
    operands, results = group_operands(graph, group), group_results(graph, group)
    cache, new, loc = operands
    (out,) = results
    heads = _literal(graph, cache, 0, "cache_write_hsd_range")
    keys = _literal(graph, cache, 1, "cache_write_hsd_range")
    depth = _literal(graph, cache, 2, "cache_write_hsd_range")
    added = _literal(graph, new, 1, "cache_write_hsd_range")
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    h, s, d, index = names.fresh("h"), names.fresh("s"), names.fresh("d"), names.fresh("index")
    lines = [f"    {variables[out]}[:] = {variables[cache]}[:]",
             f"    for {h} in range({heads}):",
             f"        for {s} in range({added}):",
             f"            {index} = {variables[loc]}[0] + {s}",
             f"            if 0 <= {index} < {keys}:",
             f"                for {d} in range({depth}):",
             f"                    {variables[out]}[{h} * {keys * depth} + {index} * {depth} + {d}] = "
             + _rounded(graph, out, f"{variables[new]}[{h} * {added * depth} + {s} * {depth} + {d}]")]
    return _kernel(name, "cache_write_hsd_range", group, operands, results, pairs, lines)


__all__ = ["cache_write_range_triton_kernel", "cache_write_range_reference_kernel"]
