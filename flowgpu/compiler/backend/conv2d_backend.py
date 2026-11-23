"""`conv2d`: tiled implicit-GEMM NHWC convolution (1x1 and small KxK)."""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier)


def _literal(graph, value, axis, what):
    extent = str(graph.values[value].shape[axis])
    if not extent.isdigit():
        raise ValueError(f"{what} requires literal extents")
    return int(extent)


def _pow2(value, minimum=16):
    return max(minimum, 1 << (max(1, value) - 1).bit_length())


def _geometry(graph, x, weight):
    h = _literal(graph, x, 1, "conv2d")
    w = _literal(graph, x, 2, "conv2d")
    channels = _literal(graph, x, 3, "conv2d")
    out_channels = _literal(graph, weight, 0, "conv2d")
    if len(graph.values[weight].shape) == 4:
        kh = _literal(graph, weight, 2, "conv2d")
        kw = _literal(graph, weight, 3, "conv2d")
    else:
        kh = kw = 1
    return h, w, channels, out_channels, kh, kw


def conv2d_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON, TritonKernel, _pointer
    assert len(group.members) == 1 and group.anchor.op == "conv2d"
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias), (out,) = operands, results
    n = _literal(graph, x, 0, "conv2d")
    h, w, channels, out_channels, kh, kw = _geometry(graph, x, weight)
    out_h, out_w = h - kh + 1, w - kw + 1
    rows = n * out_h * out_w
    bm = min(config.conv_block_m, _pow2(rows))
    bn = min(config.conv_block_n, _pow2(out_channels))
    bk = min(config.conv_block_k, _pow2(channels))
    names = Names(graph)
    pairs = [(_pointer(value), Argument("tensor", value)) for value in (x, weight, bias, out)]
    for hint, value in (("BM", bm), ("BN", bn), ("BK", bk)):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(value))))
    row_block, out_block, rows_off, ks, os_, acc = (
        names.fresh(hint) for hint in ("row_block", "out_block", "rows", "ks", "os", "acc"))
    dtype = DTYPE_TO_TRITON[graph.values[x].dtype]
    out_dtype = DTYPE_TO_TRITON[graph.values[out].dtype]
    kernel_weight = channels * kh * kw
    lines = [
        f"    {row_block} = tl.program_id(0)",
        f"    {out_block} = tl.program_id(1)",
        f"    {rows_off} = {row_block} * {bm} + tl.arange(0, {bm})",
        f"    {ks} = tl.arange(0, {bk})",
        f"    {os_} = {out_block} * {bn} + tl.arange(0, {bn})",
        f"    {acc} = tl.zeros(({bm}, {bn}), tl.float32)",
    ]
    for i in range(kh):
        for j in range(kw):
            a = names.fresh("a")
            b = names.fresh("b")
            src = (f"((({rows_off}) // ({out_h * out_w})) * {h} + "
                   f"((({rows_off}) % ({out_h * out_w})) // {out_w}) + {i}) * {w} + "
                   f"((({rows_off}) % ({out_h * out_w})) % {out_w}) + {j}")
            lines += [
                f"    {a} = tl.load({_pointer(x)} + ({src})[:, None] * {channels} + {ks}[None, :],",
                f"                  mask=({rows_off}[:, None] < {rows}) & ({ks}[None, :] < {channels}),",
                f"                  other=0.0).to({dtype})",
                f"    {b} = tl.load({_pointer(weight)} + {os_}[None, :] * {kernel_weight}"
                f" + {ks}[:, None] * {kh * kw} + {i * kw + j},",
                f"                  mask=({os_}[None, :] < {out_channels}) & ({ks}[:, None] < {channels}),",
                f"                  other=0.0).to({dtype})",
                f"    {acc} += tl.dot({a}, {b})",
            ]
    bias_tile, result = names.fresh("bias_tile"), names.fresh("result")
    lines += [
        f"    {bias_tile} = tl.load({_pointer(bias)} + {os_}, mask={os_} < {out_channels}, other=0.0)",
        f"    {result} = {acc} + {bias_tile}[None, :]",
        f"    tl.store({_pointer(out)} + {rows_off}[:, None] * {out_channels} + {os_}[None, :],",
        f"             {result}.to({out_dtype}),",
        f"             mask=({rows_off}[:, None] < {rows}) & ({os_}[None, :] < {out_channels}))",
    ]
    source = f"""# conv2d {kh}x{kw}: implicit GEMM over NHWC rows
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
""" + "\n".join(lines) + "\n"
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
    h, w, channels, out_channels, kh, kw = _geometry(graph, x, weight)
    out_h, out_w = h - kh + 1, w - kw + 1
    rows = n * out_h * out_w
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    row, o, c, i, j, total = (names.fresh(hint) for hint in
                              ("row", "o", "c", "i", "j", "total"))
    lines = [
        f"    for {row} in range({rows}):",
        f"        {i} = (({row} % {out_h * out_w}) // {out_w})",
        f"        {j} = ({row} % {out_w})",
        f"        _batch = ({row} // {out_h * out_w}) * {h}",
        f"        for {o} in range({out_channels}):",
        f"            {total} = 0.0",
        f"            for {c} in range({channels}):",
    ]
    for kh_index in range(kh):
        for kw_index in range(kw):
            x_index = (f"((_batch + ({i}) + {kh_index}) * {w} + (({j}) + {kw_index}))"
                       f" * {channels} + {c}")
            w_index = (f"({o}) * {channels * kh * kw} + ({c}) * {kh * kw}"
                       f" + {kh_index * kw + kw_index}")
            lines.append(f"                {total} += ({variables[x]}[{x_index}]"
                         f" * {variables[weight]}[{w_index}])")
    lines.append(f"            {variables[out]}[{row} * {out_channels} + {o}] = "
                 + _rounded(graph, out, f"{total} + {variables[bias]}[{o}]"))
    return _kernel(name, "conv2d", group, operands, results, pairs, lines)


__all__ = ["conv2d_triton_kernel", "conv2d_reference_kernel"]
