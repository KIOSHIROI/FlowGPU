"""`cache_write`: functional row write into a [S,H,D] KV cache."""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier)


def _literal(graph, value, axis, what):
    extent = str(graph.values[value].shape[axis])
    if not extent.isdigit():
        raise ValueError(f"{what} requires literal extents")
    return int(extent)


def cache_write_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import TritonKernel, _pointer
    assert len(group.members) == 1 and group.anchor.op == "cache_write"
    operands, results = group_operands(graph, group), group_results(graph, group)
    cache, token, loc = operands
    (out,) = results
    keys = _literal(graph, cache, 0, "cache_write")
    heads = _literal(graph, cache, 1, "cache_write")
    depth = _literal(graph, cache, 2, "cache_write")
    block = 1 << (depth - 1).bit_length()
    names = Names(graph)
    pairs = [(_pointer(value), Argument("tensor", value)) for value in (cache, token, loc, out)]
    pairs.append(("D: tl.constexpr", Argument("value", str(block))))
    row, d, target, old, new, value = (names.fresh(hint) for hint in
                                       ("row", "d", "target", "old", "new", "value"))
    source = f"""# cache_write: one row copy, optionally overwritten at loc
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
    {row} = tl.program_id(0)
    {d} = tl.arange(0, D)
    {target} = tl.load({_pointer(loc)})
    {old} = tl.load({_pointer(cache)} + {row} * {depth} + {d}, mask={d} < {depth}, other=0.0)
    {new} = tl.load({_pointer(token)} + ({row} % {heads}) * {depth} + {d}, mask={d} < {depth}, other=0.0)
    {value} = tl.where({row} // {heads} == {target}, {new}, {old})
    tl.store({_pointer(out)} + {row} * {depth} + {d}, {value}, mask={d} < {depth})
"""
    return TritonKernel(name, "cache_write", ("cache_write",), operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs), f"({keys * heads},)", source,
                        launch_options=(("num_warps", config.num_warps),))


def cache_write_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded
    operands, results = group_operands(graph, group), group_results(graph, group)
    cache, token, loc = operands
    (out,) = results
    keys = _literal(graph, cache, 0, "cache_write")
    heads = _literal(graph, cache, 1, "cache_write")
    depth = _literal(graph, cache, 2, "cache_write")
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    s, h, d = names.fresh("s"), names.fresh("h"), names.fresh("d")
    lines = [f"    {variables[out]}[:] = {variables[cache]}[:]",
             f"    if 0 <= {variables[loc]}[0] < {keys}:",
             f"        for {h} in range({heads}):",
             f"            for {d} in range({depth}):",
             f"                {variables[out]}[{variables[loc]}[0] * {heads * depth} + {h} * {depth} + {d}] = "
             + _rounded(graph, out, f"{variables[token]}[{h} * {depth} + {d}]")]
    return _kernel(name, "cache_write", group, operands, results, pairs, lines)


__all__ = ["cache_write_triton_kernel", "cache_write_reference_kernel"]
