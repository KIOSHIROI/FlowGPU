"""`conv2d_same`: small odd KxK convolution with `(K-1)//2` zero padding."""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier)
from flowgpu.compiler.backend.conv2d_backend import _literal, _pow2


def _geometry(graph, x, weight):
    n = _literal(graph, x, 0, "conv2d_same")
    h = _literal(graph, x, 1, "conv2d_same")
    w = _literal(graph, x, 2, "conv2d_same")
    channels = _literal(graph, x, 3, "conv2d_same")
    out_channels = _literal(graph, weight, 0, "conv2d_same")
    kh = _literal(graph, weight, 2, "conv2d_same")
    kw = _literal(graph, weight, 3, "conv2d_same")
    if kh != kw or kh % 2 == 0:
        raise ValueError("conv2d_same requires an odd square kernel")
    return n, h, w, channels, out_channels, kh


def conv2d_same_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON, TritonKernel, _pointer
    assert len(group.members) == 1 and group.anchor.op == "conv2d_same"
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias), (out,) = operands, results
    n, h, w, channels, out_channels, kernel = _geometry(graph, x, weight)
    pad = (kernel - 1) // 2
    rows = n * h * w
    bm = min(config.conv_block_m, _pow2(rows))
    bn = min(config.conv_block_n, _pow2(out_channels))
    bk = min(config.conv_block_k, _pow2(channels))
    names = Names(graph)
    pairs = [(_pointer(value), Argument("tensor", value)) for value in (x, weight, bias, out)]
    for hint, value in (("BM", bm), ("BN", bn), ("BK", bk)):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(value))))
    row_block, out_block, rows_off, cols, outs, acc = (
        names.fresh(hint) for hint in ("row_block", "out_block", "rows", "cols", "outs", "acc"))
    dtype = DTYPE_TO_TRITON[graph.values[x].dtype]
    out_dtype = DTYPE_TO_TRITON[graph.values[out].dtype]
    kernel_weight = channels * kernel * kernel
    lines = [
        f"    {row_block} = tl.program_id(0)",
        f"    {out_block} = tl.program_id(1)",
        f"    {rows_off} = {row_block} * {bm} + tl.arange(0, {bm})",
        f"    {cols} = tl.arange(0, {bk})",
        f"    {outs} = {out_block} * {bn} + tl.arange(0, {bn})",
        f"    {acc} = tl.zeros(({bm}, {bn}), tl.float32)",
    ]
    for i in range(kernel):
        for j in range(kernel):
            a = names.fresh("a")
            b = names.fresh("b")
            src_h = f"((({rows_off}) % ({h * w})) // {w}) + {i - pad}"
            src_w = f"((({rows_off}) % {w}) + {j - pad})"
            src = f"((({rows_off}) // ({h * w})) * {h} + ({src_h})) * {w} + ({src_w})"
            valid = ((f"({rows_off}[:, None] < {rows}) & "
                      f"(({src_h})[:, None] >= 0) & (({src_h})[:, None] < {h}) & "
                      f"(({src_w})[:, None] >= 0) & (({src_w})[:, None] < {w})"))
            lines += [
                f"    {a} = tl.load({_pointer(x)} + ({src})[:, None] * {channels} + {cols}[None, :],",
                f"                  mask={valid} & ({cols}[None, :] < {channels}),",
                f"                  other=0.0).to({dtype})",
                f"    {b} = tl.load({_pointer(weight)} + {outs}[None, :] * {kernel_weight}"
                f" + {cols}[:, None] * {kernel * kernel} + {i * kernel + j},",
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
    source = f"""# conv2d_same {kernel}x{kernel}, padding {pad}: implicit GEMM over NHWC
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
""" + "\n".join(lines) + "\n"
    grid = f"({(rows + bm - 1) // bm}, {(out_channels + bn - 1) // bn})"
    return TritonKernel(name, "conv2d_same", ("conv2d_same",), operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs), grid, source,
                        launch_options=(("num_warps", config.num_warps),))


def conv2d_same_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded
    operands, results = group_operands(graph, group), group_results(graph, group)
    (x, weight, bias), (out,) = operands, results
    n, h, w, channels, out_channels, kernel = _geometry(graph, x, weight)
    pad = (kernel - 1) // 2
    rows = n * h * w
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    row, o, c, i, j, total, sh, sw = (names.fresh(hint) for hint in
                                      ("row", "o", "c", "i", "j", "total", "sh", "sw"))
    lines = [
        f"    for {row} in range({rows}):",
        f"        {i} = (({row} % {h * w}) // {w})",
        f"        {j} = ({row} % {w})",
        f"        for {o} in range({out_channels}):",
        f"            {total} = 0.0",
        f"            for {c} in range({channels}):",
    ]
    for kh_index in range(kernel):
        for kw_index in range(kernel):
            lines += [
                f"                {sh} = {i} + {kh_index - pad}",
                f"                {sw} = {j} + {kw_index - pad}",
                f"                if 0 <= {sh} < {h} and 0 <= {sw} < {w}:",
                f"                    {total} += ({variables[x]}[((({row}) // {h * w} * {h} + {sh}) * {w} + {sw}) * {channels} + {c}]"
                f" * {variables[weight]}[{o} * {channels * kernel * kernel} + {c} * {kernel * kernel}"
                f" + {kh_index * kernel + kw_index}])",
            ]
    lines.append(f"            {variables[out]}[{row} * {out_channels} + {o}] = "
                 + _rounded(graph, out, f"{total} + {variables[bias]}[{o}]"))
    return _kernel(name, "conv2d_same", group, operands, results, pairs, lines)


__all__ = ["conv2d_same_triton_kernel", "conv2d_same_reference_kernel"]
