"""`reduce_with_index`: the index-carrying reduction over the last axis.

One structural language feature, exposed under two names because this DSL has
no string-literal mode argument:

    values, indices = argmax_last(x)
    values, indices = argmin_last(x)

They share one type rule, one lowering (this module), and one kernel kind
(`reduce_with_index`). **The audit's structural-feature denominator counts this
family as 1, not 2.**

The contract, decided once and written here rather than inherited from a
library:

1. **Last axis only.** The reduced axis is removed, so `[..., N]` produces
   `[...,]`; a rank-1 input is refused because its result would be a scalar,
   and a pipeline value is a tensor.
2. **Ties go to the smallest index.** `torch.argmax`/`torch.argmin` do not
   guarantee a tie order, so a kernel that matched torch numerically could
   still differ from it on a tie. The language fixes the order: equal values
   select the lowest column. `test/test_reduce_with_index.py` pins an
   all-equal row to index 0.
3. **NaN is never selected.** This is `topk_last`'s candidate rule. A row
   whose values are all NaN therefore has no winner and returns the
   direction's identity -- `-inf` for `argmax_last`, `+inf` for `argmin_last`
   -- with index 0. The identity is also what an empty candidate set would
   return, so there is no fourth case.
4. **+-inf follow the normal max/min rules.** Unlike `topk_last`, where -inf is
   padding and is never selected, an infinite extreme is a *real* candidate
   here: `argmax_last([1, +inf, 2])` is `(+inf, 1)` and `argmin_last` of a row
   of `-inf` selects the first one. This is stated because the two operators
   are easy to confuse.
5. **values keep the input dtype; indices are int32.**
6. **Static extents.** The last axis must be a literal, non-empty extent and
   every leading extent must be literal: the tile, the chunk loop and the grid
   are computed from them. The rejection style and messages are `topk_last`'s.

Lowering (one Triton kernel per dtype program, one launch)
----------------------------------------------------------

`out + row` for both results: a scalar base plus the program id, with no lane
term at all. That shape is chosen, not incidental, because the measured target
contracts are hard requirements:

* **No 2-D tiles.** One program per leading-row, one 1-D lane tile over the last
  axis. `vector_tile_2d` is not used.
* **No compute-then-scatter writes.** The stores are affine, exactly
  `values_ptr + row` and `indices_ptr + row`; nothing computed in the loop
  reaches an address.
* **State stays in registers, written once outside the loop.** The carried
  pair `(best, best_index)` is updated across a *static* chunk loop
  (`tl.static_range`), and the only stores sit after that loop. The chunk width
  is read from the platform record (`reduction_block_limit`, falling back to
  `default_block_size`) rather than from a platform name.

Those three properties are why `reduce_with_index` is portable: the shape that
昆仑芯 refuses (`store_inside_loop_ok=False`, `gather_load_ok=False`,
`affine_write_addresses=True`) is never emitted. `test_reduce_with_index.py`
runs the measured contract checks on the emitted source.

Why a chunk loop when the width is already a constexpr: a single
`next_power_of_2(width)` tile is unbounded in registers on a wide row, and the
platform's measured reduction budget is the existing field that says how wide a
tile that target sustains. The loop's trip count is static (a constexpr), so
the carried pair is still one register pair per program, not memory.
"""

from compiler.backend.backends import (
    Argument,
    Names,
    group_operands,
    group_results,
    identifier,
    product,
)


def _literal(graph, value, axis, what):
    extent = str(graph.values[value].shape[axis])
    if not extent.isdigit():
        raise ValueError(f"{what} requires a literal extent")
    return int(extent)


def _pow2(value):
    return 1 << (value - 1).bit_length()


def _direction(op):
    """Whether this spelling is the maximum-selecting one.

    `argmax_last` and `argmin_last` are the same operator with this boolean
    flipped; that is the only place direction exists in the lowering.  The
    identity, the comparison and the reduction primitive are derived from it.
    """
    if op == "argmax_last":
        return True
    if op == "argmin_last":
        return False
    raise AssertionError(f"not an index-carrying reduction: {op!r}")


def _dtype_expr(graph, value):
    """A dtype expression for a tile of ``value``'s declared type."""
    from compiler.backend.triton_backend import DTYPE_TO_TRITON
    return DTYPE_TO_TRITON[graph.values[value].dtype]


def _identity(maximum):
    return 'float("-inf")' if maximum else 'float("inf")'


def triton_kernel(graph, group, memory, name, config):
    """One program per leading row: a 1-D masked tile streamed in static chunks.

    `(best, best_index)` is the two-tuple carried across the chunk loop. The
    per-chunk candidate is `(tl.max/tl.min(tile), min column reaching it)`, and
    the two pairs are combined by value first and by index on equality -- the
    index tie-break is applied at every merge, not only at the end, so two
    equal maxima in different chunks still resolve to the smaller column.
    """
    from compiler.backend.triton_backend import TritonKernel
    from compiler.backend.backends import storage_of

    assert len(group.members) == 2
    assert all(node.op in {"argmax_last", "argmin_last"} for node in group.members)
    ops = {node.op for node in group.members}
    assert len(ops) == 1, "one reduction cannot be both argmax and argmin"
    op = ops.pop()
    maximum = _direction(op)

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 1 and len(results) == 2
    (x,) = operands
    values, indices = results

    # A result with no buffer in 13.7's plan is not stored; the kernel still
    # runs for the live one (a caller commonly wants only the indices).
    keep_values = storage_of(memory, values) is not None
    keep_indices = storage_of(memory, indices) is not None
    results = tuple(value for value in results
                    if storage_of(memory, value) is not None)

    width = _literal(graph, x, -1, op)
    rows = product(graph.values[x].shape[:-1])
    if not str(rows).isdigit():
        raise ValueError(f"{op} requires literal extents so the grid is static")

    # The measured per-program reduction budget is the existing platform field
    # for "how wide a tile does this target sustain". A target without a
    # measured ceiling falls back to its default block size; no platform name
    # is consulted here.
    limit = config.reduction_block_limit or config.default_block_size
    chunk = min(limit, _pow2(width))
    chunks = (width + chunk - 1) // chunk

    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr")
           for value in operands + results}
    pairs = [(ptr[value], Argument("tensor", value))
             for value in operands + results]

    def constexpr(hint, literal):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(literal))))
        return symbol

    width_symbol = constexpr("WIDTH", width)
    chunk_symbol = constexpr("CHUNK", chunk)
    chunks_symbol = constexpr("CHUNKS", chunks)

    identity = _identity(maximum)
    extreme = "max" if maximum else "min"
    compare = ">" if maximum else "<"

    row, best, best_index, chunk_name, cols, inside, tile, valid, \
        chunk_best, chunk_index, candidate, take = (
            names.fresh(hint) for hint in
            ("row", "best", "best_index", "chunk", "cols", "inside", "tile",
             "valid", "chunk_best", "chunk_index", "candidate", "take"))

    lines = [
        f"    {row} = tl.program_id(0)",
        f"    {best} = {identity}",
        f"    {best_index} = {width_symbol}",
        f"    for {chunk_name} in tl.static_range(0, {chunks_symbol}):",
        f"        {cols} = {chunk_name} * {chunk_symbol} + tl.arange(0, {chunk_symbol})",
        f"        {inside} = {cols} < {width_symbol}",
        (
            f"        {tile} = tl.load({ptr[x]} + {row} * {width_symbol} + {cols},"
            f" mask={inside}).to(tl.float32)"
        ),
        # A candidate is a loaded lane that is inside the row and not NaN. The
        # masked lanes are never stored, and this select (not `other=`)
        # establishes the candidate rule in one place -- it also keeps the
        # source inside the target contract that says a masked load's `other=`
        # is not honoured.
        f"        {valid} = {inside} & ({tile} == {tile})",
        f"        {tile} = tl.where({valid}, {tile}, {identity})",
        f"        {chunk_best} = tl.{extreme}({tile}, 0)",
        # The index of the chunk's winner: the smallest column among the
        # chunk's *candidates* whose value equals the chunk's extreme. The
        # `valid` guard is what keeps a real -inf (max) or +inf (min) from
        # being confused with the identity that NaN/masked lanes were given.
        (
            f"        {chunk_index} = tl.min(tl.where({valid} &"
            f" ({tile} == {chunk_best}), {cols}, {width_symbol}), 0)"
        ),
        f"        {candidate} = {chunk_index} < {width_symbol}",
        (
            f"        {take} = {candidate} & (({chunk_best} {compare} {best}) |"
            f" (({chunk_best} == {best}) & ({chunk_index} < {best_index})))"
        ),
        f"        {best} = tl.where({take}, {chunk_best}, {best})",
        f"        {best_index} = tl.where({take}, {chunk_index}, {best_index})",
        # An all-NaN row never updated the carried index; its sentinel is the
        # width, and the contract's answer for "no candidate" is index 0.
        f"    {best_index} = tl.where({best_index} == {width_symbol}, 0,"
        f" {best_index})",
    ]
    # Stores once, outside the loop, on scalar bases: both addresses are
    # `ptr + row`, i.e. scalar_base + affine(program index).
    if keep_values:
        lines.append(
            f"    tl.store({ptr[values]} + {row},"
            f" {best}.to({_dtype_expr(graph, values)}))")
    if keep_indices:
        lines.append(
            f"    tl.store({ptr[indices]} + {row}, {best_index}.to(tl.int32))")

    # A target that refuses a grid beyond its hardware extent cannot launch one
    # program per leading row when the leading extents are large (the generated
    # CANN ArgMax variants reach ~10^6 rows). Stride the rows inside the kernel
    # instead, using the platform's own grid ceiling: small row counts keep the
    # one-row-per-program source byte-for-byte.
    grid_expression = f"({rows},)"
    row_count = int(rows) if str(rows).isdigit() else None
    if (row_count is not None and config.max_grid_dim is not None
            and row_count > config.max_grid_dim):
        rows_symbol = constexpr("ROWS", rows)
        first_row = names.fresh("first_row")
        looped = [
            f"    {first_row} = tl.program_id(0)",
            f"    for {row} in range({first_row}, {rows_symbol},"
            f" tl.num_programs(0)):",
        ]
        looped += ["    " + line for line in lines[1:]]
        lines = looped
        grid_expression = f"({config.max_grid_dim},)"

    label = ("index-carrying reduction over the last axis; ties to the smaller "
             "index; NaN never selected; +-inf follow the max/min rules")
    source = (
        f"# {op}: {label}\n"
        f"# argmax_last and argmin_last are ONE structural feature (one lowering).\n"
        f"@triton.jit\n"
        f"def {name}({', '.join(parameter for parameter, _ in pairs)}):\n"
        + "\n".join(lines)
        + "\n"
    )
    return TritonKernel(name, "reduce_with_index", tuple(node.op for node in group.members),
                        operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs),
                        grid_expression, source,
                        launch_options=(("num_warps", config.num_warps),))


def reference_kernel(graph, group, memory, name):
    """The same reduction as an explicit Python loop, one column at a time.

    Deliberately not `max`/`min` over a slice and deliberately not a library
    call: the oracle has to be an independent construction of the semantics, so
    that a test comparing the two is a comparison and not a tautology.
    """
    from compiler.backend.backends import storage_of
    from compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    assert len(group.members) == 2
    ops = {node.op for node in group.members}
    assert len(ops) == 1, "one reduction cannot be both argmax and argmin"
    op = ops.pop()
    maximum = _direction(op)

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 1 and len(results) == 2
    (x,) = operands
    values, indices = results

    keep_values = storage_of(memory, values) is not None
    keep_indices = storage_of(memory, indices) is not None
    results = tuple(value for value in results
                    if storage_of(memory, value) is not None)

    width = _literal(graph, x, -1, op)
    rows = product(graph.values[x].shape[:-1])
    identity = "-inf" if maximum else "inf"
    compare = ">" if maximum else "<"

    names = Names(graph)
    variables = {value: names.reserve(identifier(value))
                 for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value))
             for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])

    row, best, best_index, column, current = (
        names.fresh(hint) for hint in
        ("row", "best", "best_index", "column", "current"))

    lines = [
        f"    for {row} in range({rows}):",
        f"        {best} = float('{identity}')",
        f"        {best_index} = {width}",
        f"        for {column} in range({width}):",
        f"            {current} = float({variables[x]}[{row} * {width} + {column}])",
        # NaN is not a candidate; the same rule the emitted kernel states.
        f"            if {current} != {current}:",
        f"                continue",
        (
            f"            if {best_index} == {width} or {current} {compare}"
            f" {best} or ({current} == {best} and {column} < {best_index}):"
        ),
        f"                {best} = {current}",
        f"                {best_index} = {column}",
        # No candidate at all (an all-NaN row): the identity value, index 0.
        f"        if {best_index} == {width}:",
        f"            {best} = float('{identity}')",
        f"            {best_index} = 0",
    ]
    if keep_values:
        lines.append(f"        {variables[values]}[{row}] = "
                    + _rounded(graph, values, best))
    if keep_indices:
        lines.append(f"        {variables[indices]}[{row}] = {best_index}")
    return _kernel(name, "reduce_with_index", group, operands, results, pairs, lines)
