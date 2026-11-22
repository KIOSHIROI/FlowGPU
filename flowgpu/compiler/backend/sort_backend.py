"""`sort_last`: one ascending row sort using Triton's bitonic `tl.sort`."""
from flowgpu.compiler.backend.backends import Argument, Names, group_operands, group_results, identifier, product


def _literal(graph, value, what):
    extent = str(graph.values[value].shape[-1])
    if not extent.isdigit():
        raise ValueError(f"{what} requires a literal last axis")
    return int(extent)


def _pow2(value):
    return 1 << (value - 1).bit_length()


def _dtype(graph, value):
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON
    return DTYPE_TO_TRITON[graph.values[value].dtype]


def sort_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import TritonKernel
    assert len(group.members) == 1 and group.anchor.op == "sort_last"
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    (x,) = operands
    (out,) = results
    width = _literal(graph, x, "sort_last")
    rows = product(graph.values[x].shape[:-1])
    if not str(rows).isdigit():
        raise ValueError("sort_last requires literal leading extents")
    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr") for value in operands + results}
    pairs = [(ptr[value], Argument("tensor", value)) for value in operands + results]
    lanes = names.fresh("LANES")
    pairs.append((f"{lanes}: tl.constexpr", Argument("value", str(_pow2(width)))))
    row, cols, mask, tile = (names.fresh(hint) for hint in ("row", "cols", "mask", "tile"))
    source = f"""# sort_last: ascending by value; padding is +inf and masked on store
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
    {row} = tl.program_id(0)
    {cols} = tl.arange(0, {lanes})
    {mask} = {cols} < {width}
    {tile} = tl.load({ptr[x]} + {row} * {width} + {cols}, mask={mask},
                     other=float("inf")).to(tl.float32)
    {tile} = tl.sort({tile})
    tl.store({ptr[out]} + {row} * {width} + {cols}, {tile}.to({_dtype(graph, out)}),
             mask={mask})
"""
    return TritonKernel(name, "sort", tuple(node.op for node in group.members),
                        operands, results, tuple(p for p, _ in pairs),
                        tuple(a for _, a in pairs), f"({rows},)", source,
                        launch_options=(("num_warps", config.num_warps),))


def sort_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    (x,) = operands
    (out,) = results
    width = _literal(graph, x, "sort_last")
    rows = product(graph.values[x].shape[:-1])
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    row, column, values = names.fresh("row"), names.fresh("column"), names.fresh("values")
    lines = [
        f"    for {row} in range({rows}):",
        f"        {values} = [{variables[x]}[{row} * {width} + {column}]"
        f" for {column} in range({width})]",
        f"        {values}.sort()",
        f"        for {column} in range({width}):",
        f"            {variables[out]}[{row} * {width} + {column}] = "
        + _rounded(graph, out, f"{values}[{column}]"),
    ]
    return _kernel(name, "sort", group, operands, results, pairs, lines)


__all__ = ["sort_triton_kernel", "sort_reference_kernel"]
