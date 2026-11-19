"""Dedicated one-launch lowering for `pad_cat` (padded temporal concatenation).

    out = pad_cat(x, cache, w_l, w_r, h_t, h_b, t_front, t_back)

is `F.pad(cat([cache, x], dim=2), (w_l, w_r, h_t, h_b, t_front - Tc, t_back))`.
The cached frames occupy the front of the temporal padding, so the effective
front pad is `t_front - Tc`.

Why this is a kernel kind rather than a `layout` host lowering
-------------------------------------------------------------
A layout operator is emitted as torch slicing/cat/reshape into an already
allocated slot, which the Triton host does without a grid.  That is the wrong
shape here: the composition `cat` + `F.pad` is *three* full passes over the
output-sized buffer (one to build the concatenation, one to zero the pad, one to
copy the interior), and on the benchmark this operator comes from that is
measurably three times the necessary traffic.  So this lowering emits one device
kernel that writes every output element exactly once, either the copied source
element or the zero border.

The emitted shape is the one the competition kernel was measured to need: a flat
global output index, so the store address is the index itself and there is no
wasted lane; `e // W_OUT` and the temporal decode are constexpr divides; the
border is produced by the masks rather than by a second pass.

Deliberately *not* emitted: the per-chip variants.  On 昆仑芯 both of this
kernel's loads are masked with a predicate narrower than the store's, which the
backend contracts in `contracts.py` refuse there; that target needs a
source-indexed copy with an affine store instead.  The generic form below is what
`cuda` and the international parts run, which is exactly the split the shipped
package uses.
"""

from compiler.backend.backends import (
    Argument,
    Names,
    group_operands,
    group_results,
    identifier,
)


def _pads(graph, group):
    pads = tuple(getattr(group.anchor, "attributes", None).pads)
    if len(pads) != 6:
        raise AssertionError("pad_cat requires six padding attributes")
    return pads


def triton_kernel(graph, group, memory, name, config):
    from compiler.backend.triton_backend import TritonKernel

    assert len(group.members) == 1
    assert all(node.op == "pad_cat" for node in group.members)
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 2 and len(results) == 1
    x, cache = operands
    out = results[0]
    w_l, w_r, h_t, h_b, t_front, t_back = _pads(graph, group)
    del t_back  # t_back is already folded into the output extent by infer
    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr") for value in operands + results}
    pairs = [(ptr[value], Argument("tensor", value)) for value in operands + results]

    def constexpr(hint, argument):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", argument))
        return symbol

    b_dim = constexpr("B", Argument("shape", f"{x}:0"))
    c_dim = constexpr("C", Argument("shape", f"{x}:1"))
    t_dim = constexpr("T", Argument("shape", f"{x}:2"))
    tc_dim = constexpr("TC", Argument("shape", f"{cache}:2"))
    h_dim = constexpr("H", Argument("shape", f"{x}:3"))
    w_dim = constexpr("W", Argument("shape", f"{x}:4"))
    t_out = constexpr("T_OUT", Argument("shape", f"{out}:2"))
    h_out = constexpr("H_OUT", Argument("shape", f"{out}:3"))
    w_out = constexpr("W_OUT", Argument("shape", f"{out}:4"))
    depth = constexpr("DL", Argument("value", str(t_front - int(graph.values[cache].shape[2]))))
    top = constexpr("HT", Argument("value", str(h_t)))
    left = constexpr("WL", Argument("value", str(w_l)))

    block = names.fresh("BLOCK")
    block_size = config.default_block_size
    pairs.append((f"{block}: tl.constexpr", Argument("value", str(block_size))))

    e = names.fresh("e")
    plane = names.fresh("plane")
    total = names.fresh("total")
    in_range = names.fresh("in_range")
    t_global = names.fresh("t_global")
    r = names.fresh("r")
    tt = names.fresh("tt")
    bc = names.fresh("bc")
    h = names.fresh("h")
    w = names.fresh("w")
    ts = names.fresh("ts")
    hh = names.fresh("hh")
    ww = names.fresh("ww")
    keep = names.fresh("keep")
    cache_off = names.fresh("cache_off")
    x_off = names.fresh("x_off")
    from_cache = names.fresh("from_cache")
    v_cache = names.fresh("v_cache")
    v_x = names.fresh("v_x")
    v = names.fresh("v")

    lines = [
        f"    {e} = tl.program_id(0) * {block} + tl.arange(0, {block})",
        f"    {plane} = {h_out} * {w_out}",
        f"    {total} = {b_dim} * {c_dim} * {t_out} * {plane}",
        f"    {in_range} = {e} < {total}",
        f"    {t_global} = {e} // {plane}",
        f"    {r} = {e} - {t_global} * {plane}",
        f"    {tt} = {t_global} % {t_out}",
        f"    {bc} = {t_global} // {t_out}",
        f"    {h} = {r} // {w_out}",
        f"    {w} = {r} - {h} * {w_out}",
        f"    {ts} = {tt} - {depth}",
        f"    {hh} = {h} - {top}",
        f"    {ww} = {w} - {left}",
        f"    {keep} = ({in_range} & ({hh} >= 0) & ({hh} < {h_dim}) & ({ww} >= 0)"
        f" & ({ww} < {w_dim}) & ({ts} >= 0) & ({ts} < {tc_dim} + {t_dim}))",
        f"    {cache_off} = (({bc} * {tc_dim} + {ts}) * {h_dim} + {hh}) * {w_dim} + {ww}",
        f"    {x_off} = (({bc} * {t_dim} + ({ts} - {tc_dim})) * {h_dim} + {hh}) * {w_dim} + {ww}",
        f"    {from_cache} = {ts} < {tc_dim}",
        f"    {v_cache} = tl.load({ptr[cache]} + {cache_off}, mask={keep} & {from_cache}, other=0)",
        f"    {v_x} = tl.load({ptr[x]} + {x_off}, mask={keep} & ({ts} >= {tc_dim}), other=0)",
        f"    {v} = tl.where({from_cache}, {v_cache}, {v_x})",
        f"    tl.store({ptr[out]} + {e}, {v}, mask={in_range})",
    ]

    # Every extent of this operator is a literal (infer_pad_cat refuses symbolic
    # ones, because T + t_front + t_back is not writable from a symbolic extent),
    # so the launch grid is a compile-time constant.  It must NOT be written in
    # terms of the kernel's constexpr names: those do not exist in the host
    # function that emits the launch.
    dims = [int(d) for d in graph.values[out].shape]
    total = dims[0] * dims[1] * dims[2] * dims[3] * dims[4]
    chunks = (total + block_size - 1) // block_size
    grid = f"({chunks},)"
    parameters = tuple(parameter for parameter, _ in pairs)
    source = "# pad_cat\n@triton.jit\n"
    source += f"def {name}({', '.join(parameters)}):\n" + "\n".join(lines) + "\n"
    return TritonKernel(name, "pad_cat", tuple(node.op for node in group.members),
                        operands, results, parameters,
                        tuple(argument for _, argument in pairs), grid, source,
                        launch_options=(("num_warps", config.num_warps),))


def reference_kernel(graph, group, memory, name):
    """Host-side reference: one flat-list pass, same masks as the kernel."""
    from compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    x, cache = operands
    out = results[0]
    w_l, w_r, h_t, h_b, t_front, t_back = _pads(graph, group)
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs,
                          [d for v in operands + results for d in graph.values[v].shape])

    shapes = {v: [int(d) for d in graph.values[v].shape] for v in operands + results}
    b, c, t, h, w = shapes[x]
    tc = shapes[cache][2]
    t_o, h_o, w_o = shapes[out][2], shapes[out][3], shapes[out][4]
    depth = t_front - tc
    del t_back
    lines = [
        f"    for bc in range({b} * {c}):",
        f"        for tt in range({t_o}):",
        f"            ts = tt - ({depth})",
        f"            for hh_o in range({h_o}):",
        f"                for ww_o in range({w_o}):",
        f"                    index = ((bc * {t_o} + tt) * {h_o} + hh_o) * {w_o} + ww_o",
        f"                    hh = hh_o - ({h_t})",
        f"                    ww = ww_o - ({w_l})",
        f"                    inside = (0 <= hh < {h}) and (0 <= ww < {w})"
        f" and (0 <= ts < {tc} + {t})",
        f"                    if inside and ts < {tc}:",
        f"                        source = ((bc * {tc} + ts) * {h} + hh) * {w} + ww",
        f"                        {variables[out]}[index] = {_rounded(graph, out, variables[cache] + '[source]')}",
        f"                    elif inside:",
        f"                        source = ((bc * {t} + (ts - {tc})) * {h} + hh) * {w} + ww",
        f"                        {variables[out]}[index] = {_rounded(graph, out, variables[x] + '[source]')}",
        f"                    else:",
        f"                        {variables[out]}[index] = {_rounded(graph, out, '0')}",
    ]
    return _kernel(name, "pad_cat", group, operands, results, pairs, lines)
