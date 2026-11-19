"""`gated_delta_recurrence`: the first *temporal* kernel kind.

Every other operator in this language is a point in time: it reads its inputs and
writes its outputs, and no value survives from one launch to the next. This one
is the exception the census kept pointing at -- Task18's `fused_recurrent_gdn`,
the last real expressibility gap after the value layer and the router/dispatch
families landed (2026-10-20).

The computation, exactly as the task states it, for `t = 0..T-1`:

    state      = state * exp(g_t)                      # exponential decay
    pred       = sum_k(state[v, k] * k_t[k])           # read the state through k
    correction = (v_t - pred) * beta_t                 # delta-rule correction
    state      = state + correction[v] * k_t[k]        # rank-1 outer-product update
    output_t   = sum_k(state[v, k] * q_t[k])           # read the state through q

Why this is a kernel *kind* and not just another operator row
--------------------------------------------------------------

The five steps are elementwise and reductions, all of which the language has. The
part it cannot express is the loop: `state` must stay in registers across the
whole time axis, because the alternative -- materialising `[B, T, H, V, K]` -- is
a buffer of ~10^10 bytes for a real shape, and because the output at step t is a
readout of the state *after* step t. That is a temporal construct, and this
language deliberately has no loops (Design.md 2): every loop that exists lives in
emitted device code. So the language grows one *declared* recurrence -- a kernel
kind whose emitter runs a sequential loop over the time axis -- and keeps the
"no loops in the language" position intact.

Why the operator's signature is narrower than the task's
--------------------------------------------------------

The task also takes `scale`, optional L2 normalisation of q/k, a GQA ratio, and
an optional initial state. Three are *composition*, not recurrence, and are left
in the DSL where they belong rather than baked into a kernel that would then only
be reusable by that one task:

* `scale` is one `mul` with a broadcast scalar -- a program-level choice;
* L2 norm is `div(q, sqrt(sum_axis(mul(q, q), -1) + eps))`, all of which the
  value layer added today;
* the GQA ratio (`HV = H * ratio`) needs `repeat_interleave`, which is the
  `row_repeat` gap this census lists separately as a *performance* item.

What remains is the part that is genuinely sequential, so the operator is
`gated_delta_recurrence(q, k, v, g, beta, initial_state)` requiring `HV == H`.

Consequences of being the first temporal kind
---------------------------------------------

* Its access shape is still **block**: every individual load is a contiguous run
  of `K` (or `V`) elements inside one `(batch, head, step)` row. What is unusual
  is the *time* dimension, and that is a loop rather than a lane pattern, so
  `analysis/access_shape` correctly has nothing extra to say about it. The
  temporal cost shows up in the platform contract instead, below -- which is the
  right place for it: access shape describes one access, not an iteration.
* 昆仑芯 measured **0.03x** for a loop-carried store (Task152, and the
  `store_inside_loop_ok` contract records it), and this kernel is nothing but a
  loop carried state with a store per step. So the kind is CUDA-only: it is in
  `allowed_kernel_kinds` for `cuda` and nowhere else, which is the same
  conservative position every other unmeasured lowering takes.
"""

from compiler.backend.backends import (
    Argument,
    Names,
    group_operands,
    group_results,
    identifier,
)


def _extents(graph, value):
    extents = []
    for axis, dimension in enumerate(graph.values[value].shape):
        if not str(dimension).isdigit():
            raise ValueError(
                "gated_delta_recurrence requires literal extents: the state tile "
                "is a constexpr and the time loop is unrolled by shape")
        extents.append(int(dimension))
    return extents


def triton_kernel(graph, group, memory, name, config):
    """One program per (batch, head); the state tile never leaves registers.

    `V` and `K` are the state tile's extents, so both are constexpr. The time axis
    is a runtime loop bound -- `T` is a constexpr as well (every extent here is
    literal), which lets Triton keep `state` in registers and store one output row
    per step.
    """
    from compiler.backend.backends import storage_of
    from compiler.backend.triton_backend import TritonKernel

    assert len(group.members) == 2
    assert all(node.op == "gated_delta_recurrence" for node in group.members)
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 6 and len(results) == 2
    q, k, v, gate, beta, state = operands
    output, final_state = results

    keep_output = storage_of(memory, output) is not None
    keep_final = storage_of(memory, final_state) is not None
    results = tuple(value for value in results if storage_of(memory, value) is not None)

    b_extent, t_extent, source_heads, width = _extents(graph, q)
    value_heads = _extents(graph, v)[2]
    ratio = value_heads // source_heads
    v_extent = _extents(graph, v)[-1]

    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr") for value in operands + results}
    pairs = [(ptr[value], Argument("tensor", value)) for value in operands + results]

    def constexpr(hint, literal):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(literal))))
        return symbol

    def _pow2(value):
        width_ = 1
        while width_ < value:
            width_ *= 2
        return width_

    t_symbol = constexpr("T", t_extent)
    k_symbol = constexpr("K", width)
    v_symbol = constexpr("V", v_extent)
    # tl.arange needs a power of two; a literal state extent need not be one
    # (review P1-9, 2026-10-26).  The lanes are widened and masked.
    kw_symbol = constexpr("KW", _pow2(width))
    vw_symbol = constexpr("VW", _pow2(v_extent))
    heads_symbol = constexpr("H", source_heads)
    out_heads_symbol = constexpr("HV", value_heads)

    batch = names.fresh("b")
    head = names.fresh("h")
    source_head = names.fresh("head_source")
    step = names.fresh("t")
    v_lane = names.fresh("v_lane")
    k_lane = names.fresh("k_lane")
    decay = names.fresh("decay")
    k_t = names.fresh("k_t")
    q_t = names.fresh("q_t")
    v_t = names.fresh("v_t")
    state_address = names.fresh("state_address")
    state_tile = names.fresh("state_tile")
    prediction = names.fresh("prediction")
    correction = names.fresh("correction")
    row = names.fresh("row")
    qk_base = names.fresh("qk_base")
    vo_base = names.fresh("vo_base")

    v_keep = names.fresh("v_keep")
    k_keep = names.fresh("k_keep")
    tile_keep = names.fresh("tile_keep")
    stores = []
    if keep_output:
        stores.append(f"        tl.store({ptr[output]} + {vo_base} + {v_lane}, "
                      f"{row}.to({_dtype(graph, output)}), mask={v_keep})")
    lines = [
        f"    {batch} = tl.program_id(0)",
        f"    {head} = tl.program_id(1)",
        f"    {v_lane} = tl.arange(0, {vw_symbol})",
        f"    {k_lane} = tl.arange(0, {kw_symbol})",
        f"    {v_keep} = {v_lane} < {v_symbol}",
        f"    {k_keep} = {k_lane} < {k_symbol}",
        f"    {source_head} = {head} // {ratio}",
        f"    {state_address} = ({batch} * {out_heads_symbol} + {head}) * "
        f"({v_symbol} * {k_symbol})"
        f" + {v_lane}[:, None] * {k_symbol} + {k_lane}[None, :]",
        f"    {tile_keep} = {v_keep}[:, None] & {k_keep}[None, :]",
        f"    {state_tile} = tl.load({ptr[state]} + {state_address},"
        f" mask={tile_keep}, other=0.0).to(tl.float32)",
        f"    for {step} in range({t_symbol}):",
        f"        {qk_base} = ({batch} * {t_symbol} + {step}) * {heads_symbol} * {k_symbol}"
        f" + {source_head} * {k_symbol}",
        f"        {k_t} = tl.load({ptr[k]} + {qk_base} + {k_lane}, mask={k_keep},"
        f" other=0.0).to(tl.float32)",
        f"        {q_t} = tl.load({ptr[q]} + {qk_base} + {k_lane}, mask={k_keep},"
        f" other=0.0).to(tl.float32)",
        f"        {vo_base} = ({batch} * {t_symbol} + {step}) * {out_heads_symbol} * "
        f"{v_symbol} + {head} * {v_symbol}",
        f"        {v_t} = tl.load({ptr[v]} + {vo_base} + {v_lane}, mask={v_keep},"
        f" other=0.0).to(tl.float32)",
        f"        {decay} = tl.exp(tl.load({ptr[gate]} + "
        f"({batch} * {t_symbol} + {step}) * {out_heads_symbol} + {head}).to(tl.float32))",
        f"        {state_tile} = {state_tile} * {decay}",
        f"        {prediction} = tl.sum({state_tile} * {k_t}[None, :], axis=1)",
        f"        {correction} = ({v_t} - {prediction}) * tl.load({ptr[beta]} + "
        f"({batch} * {t_symbol} + {step}) * {out_heads_symbol} + {head}).to(tl.float32)",
        f"        {state_tile} = {state_tile} + {correction}[:, None] * {k_t}[None, :]",
        f"        {row} = tl.sum({state_tile} * {q_t}[None, :], axis=1)",
    ]
    lines += stores
    if keep_final:
        lines.append(f"    tl.store({ptr[final_state]} + {state_address}, "
                     f"{state_tile}, mask={tile_keep})")

    source = ("# gated_delta_recurrence: the state tile stays in registers across T\n"
              "@triton.jit\n"
              f"def {name}({', '.join(parameter for parameter, _ in pairs)}):\n"
              + "\n".join(lines) + "\n")
    return TritonKernel(name, "recurrence", tuple(node.op for node in group.members),
                        operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs),
                        f"({b_extent}, {value_heads})", source,
                        launch_options=(("num_warps", config.num_warps),))


def _dtype(graph, value):
    from compiler.backend.triton_backend import DTYPE_TO_TRITON
    return DTYPE_TO_TRITON[graph.values[value].dtype]


def reference_kernel(graph, group, memory, name):
    """The same five steps, one time step at a time, in plain Python.

    The working state is a local list, not the input buffer: the loop mutates it,
    and mutating an operand would corrupt the caller's tensor (the Triton lowering
    keeps the tile in registers, so it has no such problem). It is copied out once
    at the end, which is also where the `final_state` result is written -- the
    first version of this lowering forgot that write and produced NaN there.
    """
    from compiler.backend.backends import storage_of
    from compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    q, k, v, gate, beta, state = operands
    output, final_state = results
    keep_output = storage_of(memory, output) is not None
    keep_final = storage_of(memory, final_state) is not None
    results = tuple(value for value in results
                    if storage_of(memory, value) is not None)

    b_extent, t_extent, source_heads, width = _extents(graph, q)
    value_heads = _extents(graph, v)[2]
    ratio = value_heads // source_heads
    v_extent = _extents(graph, v)[-1]

    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])

    batch, head, step = (names.fresh(hint) for hint in ("b", "h", "t"))
    vr, kc, at = (names.fresh(hint) for hint in ("vr", "kc", "at"))
    decay, correction, prediction = (names.fresh(hint) for hint in
                                     ("decay", "correction", "prediction"))
    work = names.fresh("state_work")
    flat = names.fresh("flat")
    total = b_extent * value_heads * v_extent * width

    # head hv of the state reads source head hv // ratio, exactly like the kernel
    source = f"({head} // {ratio})"
    state_at = (f"({batch} * {value_heads} + {head}) * ({v_extent} * {width}) + "
                f"{vr} * {width} + {kc}")
    k_at = (f"({batch} * {t_extent} + {step}) * {source_heads} * {width} + "
            f"{source} * {width} + {kc}")
    v_at = (f"({batch} * {t_extent} + {step}) * {value_heads} * {v_extent} + "
            f"{head} * {v_extent} + {vr}")
    gate_at = f"({batch} * {t_extent} + {step}) * {value_heads} + {head}"

    lines = [
        f"    {work} = [0.0] * {total}",
        f"    for {flat} in range({total}):",
        f"        {work}[{flat}] = float({variables[state]}[{flat}])",
        f"    for {batch} in range({b_extent}):",
        f"        for {head} in range({value_heads}):",
        f"            for {step} in range({t_extent}):",
        f"                {decay} = math.exp(float({variables[gate]}[{gate_at}]))",
        f"                for {vr} in range({v_extent}):",
        f"                    {prediction} = 0.0",
        f"                    for {kc} in range({width}):",
        f"                        {work}[{state_at}] *= {decay}",
        f"                        {prediction} += {work}[{state_at}] * "
        f"float({variables[k]}[{k_at}])",
        f"                    {correction} = (float({variables[v]}[{v_at}]) - "
        f"{prediction}) * float({variables[beta]}[{gate_at}])",
        f"                    for {kc} in range({width}):",
        f"                        {work}[{state_at}] += {correction} * "
        f"float({variables[k]}[{k_at}])",
    ]
    if keep_output:
        lines += [
            f"                    {prediction} = 0.0",
            f"                    for {kc} in range({width}):",
            f"                        {prediction} += {work}[{state_at}] * "
            f"float({variables[q]}[{k_at}])",
            f"                    {variables[output]}[{v_at}] = "
            + _rounded(graph, output, prediction),
        ]
    if keep_final:
        lines += [
            f"    for {flat} in range({total}):",
            f"        {variables[final_state]}[{flat}] = "
            + _rounded(graph, final_state, f"{work}[{flat}]"),
        ]
    return _kernel(name, "recurrence", group, operands, results, pairs, lines)
