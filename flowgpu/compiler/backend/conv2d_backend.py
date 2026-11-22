"""`conv2d` 1x1: one tiled `x @ weight.T + bias` kernel, NHWC in/out."""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier, product)


def _literal(graph, value, axis, what):
    extent = str(graph.values[value].shape[axis])
    if not extent.isdigit():
        raise ValueError(f"{what} requires literal extents")
    return int(extent)


def _pow2(value, minimum=16):
    return max(minimum, 1 << (max(1, value) - 1).bit_length())


def conv2d_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON, TritonKernel, _pointer
    assert len(group.members) == 1 and group.anchor.op == "conv2d"
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias), (out,) = operands, results
    n = _literal(graph, x, 0, "conv2d")
    h = _literal(graph, x, 1, "conv2d")
    w = _literal(graph, x, 2, "conv2d")
    channels = _literal(graph, x, 3, "conv2d")
    out_channels = _literal(graph, weight, 0, "conv2d")
    rows = n * h * w
    bm = min(128, _pow2(rows))
    bn = min(128, _pow2(out_channels))
    bk = min(64, _pow2(channels))
    names = Names(graph)
    pairs = [(_pointer(value), Argument("tensor", value)) for value in (x, weight, bias, out)]
    for hint, value in (("BM", bm), ("BN", bn), ("BK", bk)):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(value))))
    (row_block, out_block, rows_off, ks, os_, acc, a, b, bias_tile, result) = (
        names.fresh(hint) for hint in
        ("row_block", "out_block", "rows", "ks", "os", "acc", "a", "b", "bias_tile", "result"))
    dtype = DTYPE_TO_TRITON[graph.values[x].dtype]
    out_dtype = DTYPE_TO_TRITON[graph.values[out].dtype]
    source = f"""# conv2d 1x1: x [NHW, C] @ weight [O, C].T + bias
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
    {row_block} = tl.program_id(0)
    {out_block} = tl.program_id(1)
    {rows_off} = {row_block} * {bm} + tl.arange(0, {bm})
    {ks} = tl.arange(0, {bk})
    {os_} = {out_block} * {bn} + tl.arange(0, {bn})
    {a} = tl.load({_pointer(x)} + {rows_off}[:, None] * {channels} + {ks}[None, :],
                  mask=({rows_off}[:, None] < {rows}) & ({ks}[None, :] < {channels}),
                  other=0.0).to({dtype})
    {b} = tl.load({_pointer(weight)} + {os_}[None, :] * {channels} + {ks}[:, None],
                  mask=({os_}[None, :] < {out_channels}) & ({ks}[:, None] < {channels}),
                  other=0.0).to({dtype})
    {acc} = tl.dot({a}, {b})
    {bias_tile} = tl.load({_pointer(bias)} + {os_}, mask={os_} < {out_channels}, other=0.0)
    {result} = {acc} + {bias_tile}[None, :]
    tl.store({_pointer(out)} + {rows_off}[:, None] * {out_channels} + {os_}[None, :],
             {result}.to({out_dtype}),
             mask=({rows_off}[:, None] < {rows}) & ({os_}[None, :] < {out_channels}))
"""
    grid = f"({(rows + bm - 1) // bm}, {(out_channels + bn - 1) // bn})"
    return TritonKernel(name, "conv2d", ("conv2d",), operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs), grid, source,
                        launch_options=(("num_warps", config.num_warps),))


def conv2d_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias), (out,) = operands, results
    n = _literal(graph, x, 0, "conv2d")
    h = _literal(graph, x, 1, "conv2d")
    w = _literal(graph, x, 2, "conv2d")
    channels = _literal(graph, x, 3, "conv2d")
    out_channels = _literal(graph, weight, 0, "conv2d")
    rows = n * h * w
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    row, o, c, total = (names.fresh(hint) for hint in ("row", "o", "c", "total"))
    lines = [
        f"    for {row} in range({rows}):",
        f"        for {o} in range({out_channels}):",
        f"            {total} = 0.0",
        f"            for {c} in range({channels}):",
        f"                {total} += ({variables[x]}[{row} * {channels} + {c}]"
        f" * {variables[weight]}[{o} * {channels} + {c}])",
        f"            {variables[out]}[{row} * {out_channels} + {o}] = " + _rounded(
            graph, out, f"{total} + {variables[bias]}[{o}]"),
    ]
    return _kernel(name, "conv2d", group, operands, results, pairs, lines)


__all__ = ["conv2d_triton_kernel", "conv2d_reference_kernel"]
