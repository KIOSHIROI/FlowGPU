"""`layer_norm`: one fp32 row normalization with weight/bias and scalar eps."""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier, product,
    scalar_operands, tensor_operands)


def _literal(graph, value, what):
    extent = str(graph.values[value].shape[-1])
    if not extent.isdigit():
        raise ValueError(f"{what} requires a literal last axis")
    return int(extent)


def layer_norm_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON, TritonKernel, _pointer
    assert len(group.members) == 1 and group.anchor.op == "layer_norm"
    assert not group.epilogue, "layer_norm currently lowers without a fused epilogue"
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias, eps), (out,) = operands, results
    width = _literal(graph, x, "layer_norm")
    rows = product(graph.values[x].shape[:-1])
    if not str(rows).isdigit():
        raise ValueError("layer_norm requires literal leading extents")
    lanes = 1 << (width - 1).bit_length()
    names = Names(graph)
    pairs = []
    for value in (x, weight, bias, out):
        pairs.append((_pointer(value), Argument("tensor", value)))
    eps_name = names.fresh("eps")
    pairs.append((eps_name, Argument("scalar", eps)))
    lanes_name = names.fresh("LANES")
    pairs.append((f"{lanes_name}: tl.constexpr", Argument("value", str(lanes))))
    row, cols, mask, tile = (names.fresh(hint) for hint in ("row", "cols", "mask", "tile"))
    mean, diff, var, den = (names.fresh(hint) for hint in ("mean", "diff", "var", "den"))
    scale, shift, result = (names.fresh(hint) for hint in ("scale", "shift", "result"))
    out_dtype = DTYPE_TO_TRITON[graph.values[out].dtype]
    source = f"""# layer_norm: fp32 mean/variance, rsqrt(var + eps), then weight/bias
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
    {row} = tl.program_id(0)
    {cols} = tl.arange(0, {lanes_name})
    {mask} = {cols} < {width}
    {tile} = tl.load({_pointer(x)} + {row} * {width} + {cols}, mask={mask},
                     other=0.0).to(tl.float32)
    {mean} = tl.sum({tile}, axis=0) / {width}
    {diff} = {tile} - {mean}
    {var} = tl.sum({diff} * {diff}, axis=0) / {width}
    {den} = tl.rsqrt({var} + {eps_name})
    {scale} = tl.load({_pointer(weight)} + {cols}, mask={mask}, other=0.0).to(tl.float32)
    {shift} = tl.load({_pointer(bias)} + {cols}, mask={mask}, other=0.0).to(tl.float32)
    {result} = ({diff} * {den}) * {scale} + {shift}
    tl.store({_pointer(out)} + {row} * {width} + {cols},
             {result}.to({out_dtype}), mask={mask})
"""
    return TritonKernel(name, "layer_norm", ("layer_norm",), operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs), f"({rows},)", source,
                        launch_options=(("num_warps", config.num_warps),))


def layer_norm_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias, eps), (out,) = operands, results
    width = _literal(graph, x, "layer_norm")
    rows = product(graph.values[x].shape[:-1])
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in (x, weight, bias, out)]
    pairs.append((variables[eps], Argument("scalar", eps)))
    _dimension_parameters(names, pairs, [d for value in (x, weight, bias, out)
                                        for d in graph.values[value].shape])
    row, column, total, mean, diff, var = (names.fresh(hint) for hint in
                                           ("row", "column", "total", "mean", "diff", "var"))
    lines = [
        f"    for {row} in range({rows}):",
        f"        {total} = sum({variables[x]}[{row} * {width} + {column}]"
        f" for {column} in range({width}))",
        f"        {mean} = {total} / {width}",
        f"        {var} = sum(({variables[x]}[{row} * {width} + {column}] - {mean}) ** 2"
        f" for {column} in range({width})) / {width}",
        f"        for {column} in range({width}):",
        f"            {diff} = ({variables[x]}[{row} * {width} + {column}] - {mean})"
        f" / math.sqrt({var} + {variables[eps]})",
        f"            {variables[out]}[{row} * {width} + {column}] = " + _rounded(
            graph, out,
            f"{diff} * {variables[weight]}[{column}] + {variables[bias]}[{column}]"),
    ]
    return _kernel(name, "layer_norm", group, operands, results, pairs, lines)


__all__ = ["layer_norm_triton_kernel", "layer_norm_reference_kernel"]
