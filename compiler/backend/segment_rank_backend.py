"""`segment_rank`: the destination table a MoE dispatch is built from.

The census called this capability `index_table` and it blocked four tasks (69,
80, 84, 111). Task 69 states the computation directly:

    offset      = atomic_add(masked_m[expert], 1)
    src2dst[i]  = expert * m_max + offset
    masked_m[e] = number of pairs routed to expert e

That is a *stable rank within a segment*: each slot's destination is its expert's
base plus the number of earlier slots with the same expert. Expressed that way it
needs no atomic and no compaction primitive -- it is a counting scan -- which is
what makes it a language operator rather than a hardware trick:

    counts, positions = segment_rank(ids, num_experts, m_max)

    counts[e]     = |{ i : ids[i] == e }|
    positions[i]  = e * m_max + |{ j < i : ids[j] == e }|   for ids[i] = e >= 0
    positions[i]  = -1                                      for ids[i] < 0

Three decisions, stated rather than inherited:

1. **A negative id is padding, and its position is -1.** `scatter_by_index`
   discards out-of-bounds indices, so a dispatch written as "scatter the token
   rows to `positions`" drops padded slots for free; -1 is the value that already
   means "skip" in the task statements.
2. **The rank is *stable*: earlier slot first.** The task's atomic increment
   produces exactly this order within one expert, and an unstable order would
   change which row ends up in which dispatch slot, which the grouped GEMM
   downstream can see.
3. **`num_experts` and `m_max` are literals.** `counts` has extent `num_experts`
   and the destination space is `num_experts * m_max`, so both are part of the
   output shape; this language cannot write a computed extent.

The operator is deliberately **not** a fused "dispatch": it builds the *table*.
Composing it with `scatter_by_index` is what moves the rows, and keeping the two
apart means the table can be inspected, reused by several launches, and tested
against the task's own reference implementation.

Its access shape is a plain block: the reads are `ids[i]` in lane order and the
writes are to `counts[e]`/`positions[i]` in lane order. That matters -- it means
a target that refuses gathers and scatters can still build the table, which is the
half of the dispatch that is *computation* rather than addressing.
"""

from compiler.backend.backends import (
    Argument,
    Names,
    group_operands,
    group_results,
    identifier,
)


def _segment_literals(group):
    """``(num_experts, m_max)`` as written, carried on the node's attributes."""
    segment = tuple(getattr(group.anchor, "attributes", None).segment)
    if len(segment) != 2:
        raise AssertionError("segment_rank requires num_experts and m_max")
    return segment


def _literal(graph, value, axis, what):
    extent = str(graph.values[value].shape[axis])
    if not extent.isdigit():
        raise ValueError(f"{what} requires literal extents")
    return int(extent)


def triton_kernel(graph, group, memory, name, config):
    """Two launches: seed every position with -1, then rank each expert.

    The seeding launch exists because padding has no expert: a slot with a
    negative id belongs to no program in the per-expert grid, so somebody has to
    own its -1. The protocol already allows a lowering to be several kernels
    (that is how `scatter_by_index` gets its copy), so this is one operator with
    two launches rather than one launch with a race.

    Each expert program walks every slot in a bounded chunk, keeps a running
    count across chunks and uses `tl.cumsum` inside one, which is exactly the
    "number of earlier slots with this expert" the task's atomic produces.
    """
    from compiler.backend.triton_backend import TritonKernel
    from compiler.backend.backends import product

    assert len(group.members) == 2
    assert all(node.op == "segment_rank" for node in group.members)
    from compiler.backend.backends import storage_of

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    (ids,) = operands
    counts, positions = results
    # A multi-result operator may be asked for only one of its results; the dead
    # one has no buffer and must not become a kernel argument or a store
    # (review P1-6, 2026-10-26).
    keep_counts = storage_of(memory, counts) is not None
    keep_positions = storage_of(memory, positions) is not None
    live_results = tuple(value for value in results
                         if storage_of(memory, value) is not None)

    slots = _literal(graph, ids, 0, "segment_rank")
    experts = _literal(graph, counts, 0, "segment_rank")
    m_max = _segment_literals(group)[1]
    block = config.default_block_size

    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr")
           for value in operands + live_results}
    pairs = [(ptr[value], Argument("tensor", value))
             for value in operands + live_results]

    def constexpr(hint, literal):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(literal))))
        return symbol

    experts_symbol = constexpr("EXPERTS", experts)
    slots_symbol = constexpr("SLOTS", slots)
    block_symbol = names.fresh("BLOCK")
    pairs.append((f"{block_symbol}: tl.constexpr", Argument("value", str(block))))
    m_max_symbol = constexpr("M_MAX", m_max)

    # launch 1: every destination starts at "no destination"
    flat = names.fresh("flat")
    inside = names.fresh("inside")
    reset_lines = [
        f"    {flat} = tl.program_id(0) * {block_symbol} + tl.arange(0, {block_symbol})",
        f"    {inside} = {flat} < {slots_symbol}",
    ]
    # the reset launch only exists to seed `positions`; without that result it
    # would be a kernel that writes nothing
    if keep_positions:
        reset_lines.append(
            f"    tl.store({ptr[positions]} + {flat}, -1, mask={inside})")
    else:
        reset_lines.append("    pass")
    reset_chunks = (slots + block - 1) // block
    reset_parameters = tuple(parameter for parameter, _ in pairs)
    reset_source = "# segment_rank (reset: padded slots belong to no expert)\n@triton.jit\n"
    reset_source += (f"def {name}_reset({', '.join(reset_parameters)}):\n"
                     + "\n".join(reset_lines) + "\n")
    reset_kernel = TritonKernel(f"{name}_reset", "index_table",
                                tuple(node.op for node in group.members),
                                operands, live_results, reset_parameters,
                                tuple(argument for _, argument in pairs),
                                f"({reset_chunks},)", reset_source,
                                launch_options=(("num_warps", config.num_warps),))

    # launch 2: one program per expert, ranking its slots in arrival order
    expert = names.fresh("expert")
    start = names.fresh("start")
    step = names.fresh("step")
    lanes = names.fresh("lanes")
    mine = names.fresh("mine")
    prefix = names.fresh("prefix")
    running = names.fresh("running")
    slot_id = names.fresh("slot_id")
    offset = names.fresh("offset")
    # A 64-bit id must stay 64-bit, or a wrapped value is counted for the wrong
    # expert (review P1-7, 2026-10-26).
    index_dtype = "tl.int64" if graph.values[ids].dtype == "int64" else "tl.int32"
    rank_lines = [
        f"    {expert} = tl.program_id(0)",
        f"    {start} = tl.zeros((), tl.int32)",
        f"    for {step} in range(tl.cdiv({slots_symbol}, {block_symbol})):",
        f"        {lanes} = {step} * {block_symbol} + tl.arange(0, {block_symbol})",
        f"        {inside} = {lanes} < {slots_symbol}",
        f"        {slot_id} = tl.load({ptr[ids]} + {lanes}, mask={inside}, other=-1).to({index_dtype})",
        f"        {mine} = (({slot_id} == {expert}) & {inside}).to(tl.int32)",
        f"        {prefix} = tl.cumsum({mine}, 0) - {mine}",
        f"        {offset} = {expert} * {m_max_symbol} + {start} + {prefix}",
    ]
    if keep_positions:
        rank_lines += [
            f"        tl.store({ptr[positions]} + {lanes}, {offset},",
            f"                 mask=(({slot_id} == {expert}) & {inside}))",
        ]
    rank_lines += [
        f"        {running} = {start} + tl.sum({mine}, 0)",
        f"        {start} = {running}.to(tl.int32)",
    ]
    if keep_counts:
        rank_lines += [
            f"    tl.store({ptr[counts]} + {expert}, {start}.to(tl.int32))",
        ]
    rank_parameters = tuple(parameter for parameter, _ in pairs)
    rank_source = "# segment_rank (rank: stable within each expert)\n@triton.jit\n"
    rank_source += (f"def {name}_rank({', '.join(rank_parameters)}):\n"
                    + "\n".join(rank_lines) + "\n")
    rank_kernel = TritonKernel(f"{name}_rank", "index_table",
                               tuple(node.op for node in group.members),
                               operands, live_results, rank_parameters,
                               tuple(argument for _, argument in pairs),
                               # the grid is a *host* expression: `experts` is a
                               # literal here, while `experts_symbol` only exists
                               # inside the kernel signature (caught on the H100)
                               f"({experts},)", rank_source,
                               launch_options=(("num_warps", config.num_warps),))
    return (reset_kernel, rank_kernel)


def reference_kernel(graph, group, memory, name):
    """The two-pass form as plain Python: count first, then rank in slot order."""
    from compiler.backend.reference import _dimension_parameters, _kernel

    from compiler.backend.backends import storage_of

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    (ids,) = operands
    counts, positions = results
    keep_counts = storage_of(memory, counts) is not None
    keep_positions = storage_of(memory, positions) is not None
    live_results = tuple(value for value in results
                         if storage_of(memory, value) is not None)
    slots = _literal(graph, ids, 0, "segment_rank")
    experts = _literal(graph, counts, 0, "segment_rank")
    m_max = _segment_literals(group)[1]
    names = Names(graph)
    variables = {value: names.reserve(identifier(value))
                 for value in operands + live_results}
    pairs = [(variables[value], Argument("tensor", value))
             for value in operands + live_results]
    _dimension_parameters(names, pairs, [d for value in operands + live_results
                                        for d in graph.values[value].shape])
    slot, expert, running = (names.fresh(hint) for hint in ("slot", "expert", "running"))
    lines = []
    if keep_positions:
        # every destination starts at "no destination", exactly like the reset
        # launch: a padded slot belongs to no expert, and an id outside
        # [0, num_experts) belongs to no expert either
        lines += [
            f"    for {slot} in range({slots}):",
            f"        {variables[positions]}[{slot}] = -1",
        ]
    lines += [f"    for {expert} in range({experts}):", f"        {running} = 0"]
    if keep_positions:
        lines += [f"        for {slot} in range({slots}):",
                  f"            if {variables[ids]}[{slot}] == {expert}:",
                  f"                {variables[positions]}[{slot}] = {expert} * {m_max} + {running}",
                  f"                {running} += 1"]
    else:
        # counting only: the loop still has to count this expert's slots
        lines += [f"        for {slot} in range({slots}):",
                  f"            if {variables[ids]}[{slot}] == {expert}:",
                  f"                {running} += 1"]
    if keep_counts:
        lines.append(f"        {variables[counts]}[{expert}] = {running}")
    return _kernel(name, "index_table", group, operands, live_results, pairs, lines)
