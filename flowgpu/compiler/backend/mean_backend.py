"""`mean_last`: one fp32 row mean per program, stored as a row scalar."""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier, product)


def _literal(graph, value, what):
    extent = str(graph.values[value].shape[-1])
    if not extent.isdigit():
        raise ValueError(f"{what} requires a literal last axis")
    return int(extent)


def mean_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON, TritonKernel
    assert len(group.members) == 1 and group.anchor.op == "mean_last"
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x,), (out,) = operands, results
    width = _literal(graph, x, "mean_last")
    rows = product(graph.values[x].shape[:-1])
    if not str(rows).isdigit():
        raise ValueError("mean_last requires literal leading extents")
    lanes = 1 << (width - 1).bit_length()
    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr") for value in operands + results}
    pairs = [(ptr[value], Argument("tensor", value)) for value in operands + results]
    lanes_name = names.fresh("LANES")
    pairs.append((f"{lanes_name}: tl.constexpr", Argument("value", str(lanes))))
    row, cols, mask, tile, total = (names.fresh(hint) for hint in
                                    ("row", "cols", "mask", "tile", "total"))
    out_dtype = DTYPE_TO_TRITON[graph.values[out].dtype]
    source = f"""# mean_last: fp32 sum divided by the literal row width
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
    {row} = tl.program_id(0)
    {cols} = tl.arange(0, {lanes_name})
    {mask} = {cols} < {width}
    {tile} = tl.load({ptr[x]} + {row} * {width} + {cols}, mask={mask},
                     other=0.0).to(tl.float32)
    {total} = tl.sum({tile}, axis=0) / {width}
    tl.store({ptr[out]} + {row}, {total}.to({out_dtype}))
"""
    return TritonKernel(name, "mean", ("mean_last",), operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs), f"({rows},)", source,
                        launch_options=(("num_warps", config.num_warps),))


def mean_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x,), (out,) = operands, results
    width = _literal(graph, x, "mean_last")
    rows = product(graph.values[x].shape[:-1])
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    row, column, total = names.fresh("row"), names.fresh("column"), names.fresh("total")
    lines = [
        f"    for {row} in range({rows}):",
        f"        {total} = sum({variables[x]}[{row} * {width} + {column}]"
        f" for {column} in range({width})) / {width}",
        f"        {variables[out]}[{row}] = " + _rounded(graph, out, total),
    ]
    return _kernel(name, "mean", group, operands, results, pairs, lines)


__all__ = ["mean_triton_kernel", "mean_reference_kernel"]
