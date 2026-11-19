"""Dedicated one-launch lowering for `patchify`: a static element permutation.

    out = patchify(x, ph, pw)   ==   b c (h ph) (w pw) -> b (h w) (c ph pw)

This is a *declarative* layout operator, the same species as `reshape` and
`pad_cat`: the index map is fixed by the two literal patch extents. What makes it
worth a kernel kind of its own is the access shape, not the arithmetic -- the
store walks the output densely while the load's address is a permutation of it.

That is deliberate. On 2026-10-20 the Task154 operator, which is a normalize
followed by exactly this rearrangement, measured 0.02x on 昆仑芯 in its
output-indexed form and Correctness Failed in a 2-D masked form. Emitting the
natural (contiguous store, permuted load) form makes the plan's access shape
*visible to the compiler before any text exists*: `analysis/access_shape` reports
it as `PERMUTED`, and `access_shape.platform_conflicts` refuses it on that target
with the reason the official result demonstrates. A lowering that cannot be
judged is worth less than one that can.
"""

from flowgpu.compiler.backend.backends import (
    Argument,
    Names,
    group_operands,
    group_results,
    identifier,
)


def _extents(graph, value):
    return [int(d) for d in graph.values[value].shape]


def _pads(graph, group):
    """The two patched extents, carried on the node's attributes."""
    pads = tuple(getattr(group.anchor, "attributes", None).pads)
    if len(pads) != 2:
        raise AssertionError("patchify requires two patch extents")
    return pads


def triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import TritonKernel

    assert len(group.members) == 1
    assert all(node.op == "patchify" for node in group.members)
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 1 and len(results) == 1
    x, = operands
    out = results[0]
    ph, pw = _pads(graph, group)
    b, c, h, w = _extents(graph, x)
    _, patches, lanes = _extents(graph, out)
    nh, nw = h // ph, w // pw
    names = Names(graph)
    x_ptr = names.fresh(identifier(x) + "_ptr")
    out_ptr = names.fresh(identifier(out) + "_ptr")
    pairs = [(x_ptr, Argument("tensor", x)), (out_ptr, Argument("tensor", out))]

    def constexpr(hint, value):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(value))))
        return symbol

    dims = {
        "C": c, "H": h, "W": w, "PH": ph, "PW": pw,
        "LANES": lanes, "N": patches,
    }
    symbols = {key: constexpr(key, value) for key, value in dims.items()}
    block = names.fresh("BLOCK")
    block_size = config.default_block_size
    pairs.append((f"{block}: tl.constexpr", Argument("value", str(block_size))))

    total = b * patches * lanes
    e = names.fresh("e")
    in_range = names.fresh("in_range")
    lane = names.fresh("lane")
    rest = names.fresh("rest")
    patch = names.fresh("patch")
    channel = names.fresh("channel")
    within = names.fresh("within")
    row = names.fresh("row")
    col = names.fresh("col")
    ph_index = names.fresh("ph_index")
    pw_index = names.fresh("pw_index")
    src = names.fresh("src")
    value = names.fresh("value")
    lines = [
        f"    {e} = tl.program_id(0) * {block} + tl.arange(0, {block})",
        f"    {in_range} = {e} < {total}",
        f"    {lane} = {e} % {symbols['LANES']}",
        f"    {rest} = {e} // {symbols['LANES']}",
        f"    {patch} = {rest} % {symbols['N']}",
        f"    {channel} = {lane} // ({symbols['PH']} * {symbols['PW']})",
        f"    {within} = {lane} % ({symbols['PH']} * {symbols['PW']})",
        f"    {row} = {within} // {symbols['PW']}",
        f"    {col} = {within} % {symbols['PW']}",
        f"    {ph_index} = {patch} // {nw}",
        f"    {pw_index} = {patch} % {nw}",
        f"    {src} = (((( {rest} // {symbols['N']} ) * {symbols['C']} + {channel})"
        f" * {symbols['H']} + {ph_index} * {symbols['PH']} + {row})"
        f" * {symbols['W']} + {pw_index} * {symbols['PW']} + {col})",
        f"    {value} = tl.load({x_ptr} + {src}, mask={in_range}, other=0)",
        f"    tl.store({out_ptr} + {e}, {value}, mask={in_range})",
    ]
    parameters = tuple(parameter for parameter, _ in pairs)
    source = "# patchify\n@triton.jit\n"
    source += f"def {name}({', '.join(parameters)}):\n" + "\n".join(lines) + "\n"
    chunks = (total + block_size - 1) // block_size
    return TritonKernel(name, "patchify", tuple(node.op for node in group.members),
                        operands, results, parameters,
                        tuple(argument for _, argument in pairs),
                        f"({chunks},)", source,
                        launch_options=(("num_warps", config.num_warps),))


def reference_kernel(graph, group, memory, name):
    """Host-side reference: the same index map, written out directly."""
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    x, = operands
    out = results[0]
    ph, pw = _pads(graph, group)
    b, c, h, w = _extents(graph, x)
    _, patches, lanes = _extents(graph, out)
    nh, nw = h // ph, w // pw
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs,
                          [d for v in operands + results for d in graph.values[v].shape])
    lines = [
        f"    for batch in range({b}):",
        f"        for patch in range({patches}):",
        f"            for lane in range({lanes}):",
        f"                channel = lane // ({ph} * {pw})",
        f"                within = lane % ({ph} * {pw})",
        f"                index = (batch * {patches} + patch) * {lanes} + lane",
        f"                source = (((batch * {c} + channel) * {h}"
        f" + (patch // {nw}) * {ph} + within // {pw}) * {w}"
        f" + (patch % {nw}) * {pw} + within % {pw})",
        f"                {variables[out]}[index] = "
        f"{_rounded(graph, out, variables[x] + '[source]')}",
    ]
    return _kernel(name, "patchify", group, operands, results, pairs, lines)
