"""Native Triton attention for the 2-D/4-D causal and non-causal forms.

One program per (batch, query head, query position): scores are a masked
dot-product against the whole key axis, softmax runs in registers, and the
value projection reuses the same weights. GQA reads key/value head
``h // (H // G)``. The library path remains available via
``gpu_backend='hybrid'``.
"""
from flowgpu.compiler.backend.backends import (
    Argument, Names, group_operands, group_results, identifier, product)


def _literal(shape, axis, what):
    extent = str(shape[axis])
    if not extent.isdigit():
        raise ValueError(f"{what} requires literal extents")
    return int(extent)


def _pow2(value):
    return 1 << (max(1, value) - 1).bit_length()


def attention_triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON, TritonKernel, _pointer
    anchor = group.anchor
    if anchor.op not in {"attention", "causal_attention", "window_attention"}:
        raise ValueError("not an attention anchor")
    operands, results = group_operands(graph, group), group_results(graph, group)
    windowed = anchor.op == "window_attention"
    q, k, v = operands[:3]
    (out,) = results
    q_shape = graph.values[q].shape
    k_shape = graph.values[k].shape
    v_shape = graph.values[v].shape
    rank = len(q_shape)
    if rank == 2:
        batch = heads = groups = 1
        length, q_depth = _literal(q_shape, 0, "attention"), _literal(q_shape, 1, "attention")
        keys, k_depth = _literal(k_shape, 0, "attention"), _literal(k_shape, 1, "attention")
        width = _literal(v_shape, 1, "attention")
    elif rank == 3:
        batch = 1
        heads, length, q_depth = (_literal(q_shape, axis, "attention") for axis in range(3))
        groups, keys, k_depth = (_literal(k_shape, axis, "attention") for axis in range(3))
        width = _literal(v_shape, 2, "attention")
    else:
        batch, heads, length, q_depth = (_literal(q_shape, axis, "attention") for axis in range(4))
        groups, keys, k_depth = (_literal(k_shape, axis, "attention") for axis in (1, 2, 3))
        width = _literal(v_shape, 3, "attention")
    if q_depth != k_depth:
        raise ValueError("attention q/k depth mismatch")
    if keys < length and anchor.op == "causal_attention":
        raise ValueError("causal attention requires the key length to be at least the query length")
    ratio = heads // groups
    block_s, block_d, block_v = _pow2(keys), _pow2(q_depth), _pow2(width)
    rows = batch * heads * length
    names = Names(graph)
    pairs = [(_pointer(value), Argument("tensor", value)) for value in (q, k, v, out)]
    if windowed:
        pairs.append((_pointer(operands[3]), Argument("tensor", operands[3])))
    for hint, value in (("BS", block_s), ("BD", block_d), ("BV", block_v)):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(value))))
    (row, b_, h_, i, js, d, c, qt, kt, vt, scores, mask, weights, total) = (
        names.fresh(hint) for hint in
        ("row", "b", "h", "i", "js", "d", "c", "qt", "kt", "vt", "scores", "mask", "weights", "total"))
    if rank == 4:
        kbase = f"{b_} * {groups} * {keys} * {k_depth} + ({h_} // {ratio}) * {keys} * {k_depth}"
        vbase = f"{b_} * {groups} * {keys} * {width} + ({h_} // {ratio}) * {keys} * {width}"
        qbase = f"{b_} * {heads} * {length} * {q_depth} + {h_} * {length} * {q_depth} + {i} * {q_depth}"
        obase = f"{b_} * {heads} * {length} * {width} + {h_} * {length} * {width} + {i} * {width}"
    elif rank == 3:
        kbase = f"({h_} // {ratio}) * {keys} * {k_depth}"
        vbase = f"({h_} // {ratio}) * {keys} * {width}"
        qbase = f"{h_} * {length} * {q_depth} + {i} * {q_depth}"
        obase = f"{h_} * {length} * {width} + {i} * {width}"
    else:
        kbase = "0"
        vbase = "0"
        qbase = f"{i} * {q_depth}"
        obase = f"{i} * {width}"
    scale = f"{float(q_depth) ** 0.5:.17g}"
    causal = anchor.op == "causal_attention"
    window = f"({i} + {keys} - {length})" if causal else f"({keys} - 1)"
    keep = f"({js} <= {window})" if causal else f"({js} < {keys})"
    vlen = names.fresh("vlen") if windowed else None
    vlen_index = b_ if rank == 4 else "0"
    if windowed:
        keep = f"({js} < {vlen})"
    lines = [
        f"    {row} = tl.program_id(0)",
        f"    {b_} = {row} // ({heads} * {length})",
        f"    {h_} = ({row} // {length}) % {heads}",
        f"    {i} = {row} % {length}",
        f"    {js} = tl.arange(0, BS)",
        f"    {d} = tl.arange(0, BD)",
        f"    {c} = tl.arange(0, BV)",
        (f"    {vlen} = tl.load({_pointer(operands[3])} + {vlen_index})" if windowed else ""),
        f"    {qt} = tl.load({_pointer(q)} + {qbase} + {d}, mask={d} < {q_depth}, other=0.0).to(tl.float32)",
        f"    {kt} = tl.load({_pointer(k)} + {kbase} + {js}[:, None] * {k_depth} + {d}[None, :],",
        f"                  mask=({js}[:, None] < {keys}) & ({d}[None, :] < {k_depth}), other=0.0).to(tl.float32)",
        f"    {scores} = tl.sum({kt} * {qt}[None, :], axis=1) / {scale}",
        f"    {mask} = ({keep}) & ({js} < {keys})",
        f"    {scores} = tl.where({mask}, {scores}, -float(\"inf\"))",
        f"    {weights} = tl.exp({scores} - tl.max({scores}))",
        f"    {total} = tl.sum({weights})",
        f"    {vt} = tl.load({_pointer(v)} + {vbase} + {js}[:, None] * {width} + {c}[None, :],",
        f"                  mask=({js}[:, None] < {keys}) & ({c}[None, :] < {width}), other=0.0).to(tl.float32)",
        f"    {row} = tl.sum({weights}[:, None] * {vt}, axis=0) / {total}",
        f"    tl.store({_pointer(out)} + {obase} + {c}, {row}.to({DTYPE_TO_TRITON[graph.values[out].dtype]}),",
        f"             mask={c} < {width})",
    ]
    source = f"""# native {'causal ' if causal else ''}attention (GQA ratio {ratio})
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
""" + "\n".join(lines) + "\n"
    lines = [line for line in lines if line]
    return TritonKernel(name, "attention", (anchor.op,), operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs), f"({rows},)", source,
                        launch_options=(("num_warps", config.num_warps),))


__all__ = ["attention_triton_kernel"]
