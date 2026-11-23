"""`conv2d_stride2`: small KxK stride-2 convolution, no padding."""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier)
from flowgpu.compiler.backend.conv2d_backend import _literal, _pow2


def _geometry(graph, x, weight):
    n = _literal(graph, x, 0, "conv2d_stride2")
    h = _literal(graph, x, 1, "conv2d_stride2")
    w = _literal(graph, x, 2, "conv2d_stride2")
    channels = _literal(graph, x, 3, "conv2d_stride2")
    out_channels = _literal(graph, weight, 0, "conv2d_stride2")
    kh = _literal(graph, weight, 2, "conv2d_stride2")
    kw = _literal(graph, weight, 3, "conv2d_stride2")
    return n, h, w, channels, out_channels, kh, kw


def _out_extent(extent, kernel):
    return (extent - kernel) // 2 + 1


def conv2d_stride2_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON, TritonKernel, _pointer
    assert len(group.members) == 1 and group.anchor.op == "conv2d_stride2"
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias), (out,) = operands, results
    n, h, w, channels, out_channels, kh, kw = _geometry(graph, x, weight)
    out_h = _out_extent(h, kh)
    out_w = _out_extent(w, kw)
    plane = out_h * out_w
    rows = n * plane
    bm = min(128, _pow2(rows))
    bn = min(128, _pow2(out_channels))
    bk = min(64, _pow2(channels))
    names = Names(graph)
    pairs = [(_pointer(value), Argument("tensor", value)) for value in (x, weight, bias, out)]
    for hint, value in (("BM", bm), ("BN", bn), ("BK", bk)):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(value))))
    row_block, out_block, rows_off, cols, outs, acc = (
        names.fresh(hint) for hint in ("row_block", "out_block", "rows", "cols", "outs", "acc"))
    dtype = DTYPE_TO_TRITON[graph.values[x].dtype]
    out_dtype = DTYPE_TO_TRITON[graph.values[out].dtype]
    kernel_weight = channels * kh * kw
    lines = [
        f"    {row_block} = tl.program_id(0)",
        f"    {out_block} = tl.program_id(1)",
        f"    {rows_off} = {row_block} * {bm} + tl.arange(0, {bm})",
        f"    {cols} = tl.arange(0, {bk})",
        f"    {outs} = {out_block} * {bn} + tl.arange(0, {bn})",
        f"    {acc} = tl.zeros(({bm}, {bn}), tl.float32)",
    ]
    for i in range(kh):
        for j in range(kw):
            a = names.fresh("a")
            b = names.fresh("b")
            src = (f"((({rows_off}) // {plane}) * {h} + "
                   f"((({rows_off}) % {plane}) // {out_w}) * 2 + {i}) * {w} + "
                   f"((({rows_off}) % {out_w}) * 2 + {j})")
            lines += [
                f"    {a} = tl.load({_pointer(x)} + ({src})[:, None] * {channels} + {cols}[None, :],",
                f"                  mask=({rows_off}[:, None] < {rows}) & ({cols}[None, :] < {channels}),",
                f"                  other=0.0).to({dtype})",
                f"    {b} = tl.load({_pointer(weight)} + {outs}[None, :] * {kernel_weight}"
                f" + {cols}[:, None] * {kh * kw} + {i * kw + j},",
                f"                  mask=({outs}[None, :] < {out_channels}) & ({cols}[:, None] < {channels}),",
                f"                  other=0.0).to({dtype})",
                f"    {acc} += tl.dot({a}, {b})",
            ]
    bias_tile, result = names.fresh("bias_tile"), names.fresh("result")
    lines += [
        f"    {bias_tile} = tl.load({_pointer(bias)} + {outs}, mask={outs} < {out_channels}, other=0.0)",
        f"    {result} = {acc} + {bias_tile}[None, :]",
        f"    tl.store({_pointer(out)} + {rows_off}[:, None] * {out_channels} + {outs}[None, :],",
        f"             {result}.to({out_dtype}),",
        f"             mask=({rows_off}[:, None] < {rows}) & ({outs}[None, :] < {out_channels}))",
    ]
    source = f"""# conv2d_stride2 {kh}x{kw}: implicit GEMM, stride 2, NHWC
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
""" + "\n".join(lines) + "\n"
    grid = f"({(rows + bm - 1) // bm}, {(out_channels + bn - 1) // bn})"
    return TritonKernel(name, "conv2d_stride2", ("conv2d_stride2",), operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs), grid, source,
                        launch_options=(("num_warps", config.num_warps),))


def conv2d_stride2_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias), (out,) = operands, results
    n, h, w, channels, out_channels, kh, kw = _geometry(graph, x, weight)
    out_h, out_w = _out_extent(h, kh), _out_extent(w, kw)
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
        f"        for {o} in range({out_channels}):",
        f"            {total} = 0.0",
        f"            for {c} in range({channels}):",
    ]
    for kh_index in range(kh):
        for kw_index in range(kw):
            x_index = (f"((({row}) // {out_h * out_w} * {h} + ({i}) * 2 + {kh_index}) * {w}"
                       f" + ({j}) * 2 + {kw_index}) * {channels} + {c}")
            w_index = (f"({o}) * {channels * kh * kw} + ({c}) * {kh * kw}"
                       f" + {kh_index * kw + kw_index}")
            lines.append(f"                {total} += ({variables[x]}[{x_index}]"
                         f" * {variables[weight]}[{w_index}])")
    lines.append(f"            {variables[out]}[{row} * {out_channels} + {o}] = "
                 + _rounded(graph, out, f"{total} + {variables[bias]}[{o}]"))
    return _kernel(name, "conv2d_stride2", group, operands, results, pairs, lines)


__all__ = ["conv2d_stride2_triton_kernel", "conv2d_stride2_reference_kernel"]
