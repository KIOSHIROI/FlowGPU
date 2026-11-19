"""`topk_last` and `gather_last`: which of these is biggest, and fetch by it.

The content census put this family first among the remaining gaps (10 tasks:
25, 26, 27, 31, 38, 44, 61, 70, 85, 108). The router/gate shape needs *both*
halves and neither alone is enough:

    logits = matmul(x, router_weight)          # [B, E]
    ids    = topk_last(logits, k)              # [B, k] int32
    probs  = softmax(logits)                   # [B, E]
    weights = gather_last(probs, ids)          # [B, k]  <- weights from the *full*
                                               #            softmax, not a renormalised
                                               #            top-k softmax

That second line is why `gather_last` is not optional. The MoE router tasks are
explicit that the weights come from the full-E softmax ("weights come from the
global softmax, not renormalised over the selected topk"), so a `softmax` over a
top-k slice is a different function. A last-axis gather by an index tensor has no
formulation in terms of the axis-0 row gather the language has, because the row
is selected by a value *inside* the row.

Three semantic decisions are made here, not discovered later
------------------------------------------------------------

1. **Ties go to the lower column index.** `torch.argsort(...)[:, :k]` does not
   specify tie order, so a kernel that matched it numerically could still differ
   from it on a tie. The language fixes the order instead of inheriting an
   unspecified one: descending by value, and for equal values the smaller column
   first. `test_topk_family.py` pins it with a row of duplicate values.
2. **NaN is never selected.** `torch.topk` puts NaN *first*; `>` comparisons make
   NaN sorts-last naturally, and a router whose logits went NaN selects nothing
   rather than training on a NaN. This is the opposite of torch, so it is stated
   here and tested rather than discovered by a user.
3. **`k` is a static literal.** The output extent is `[..., k]`, and this
   language cannot write a computed output extent (Design.md 5), so `k` is a
   literal in the same way `pad_cat`'s pads and `patchify`'s patch size are.

`gather_last` keeps `gather_rows`' boundary convention: a negative index is
normalised once by adding the extent, and an index still out of bounds reads as
zero. Out-of-bounds *writes* do not exist here because the operator only reads.
"""

from flowgpu.compiler.backend.backends import (
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


def _dtype_expr(graph, value):
    """A dtype expression for a tile of ``value``'s declared type."""
    from flowgpu.compiler.backend.triton_backend import DTYPE_TO_TRITON
    return DTYPE_TO_TRITON[graph.values[value].dtype]


def topk_triton_kernel(graph, group, memory, name, config):
    """One program per row: k rounds of argmax, each selecting the lower column.

    `tl.max` over the row gives the value; the index is the *smallest* column
    whose value equals it, which is what makes the tie rule the language's rather
    than whatever order a reduction tree happens to produce. The selected lane is
    then set to -inf and the next round runs, so no sort is needed for a k this
    language can write (a literal, and small in every task that needs one).

    A row with fewer than k candidates pads with value 0 and index 0, and the
    emitted source says so explicitly (`best == -inf` means no candidate left)
    rather than letting a -inf reach the caller as if it were a logit.
    """
    from flowgpu.compiler.backend.triton_backend import TritonKernel

    assert len(group.members) == 2
    assert all(node.op == "topk_last" for node in group.members)
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 1 and len(results) == 2
    (x,) = operands
    values, indices = results

    # A router needs the indices and not the values; a result with no buffer in
    # 13.7's plan is not stored, and the kernel still runs for the live one.
    from flowgpu.compiler.backend.backends import storage_of
    keep_values = storage_of(memory, values) is not None
    keep_indices = storage_of(memory, indices) is not None
    results = tuple(value for value in results
                    if storage_of(memory, value) is not None)

    width = _literal(graph, x, -1, "topk_last")
    k = _literal(graph, values, -1, "topk_last")
    if k > width:
        raise ValueError(f"topk_last k={k} exceeds the last axis ({width})")
    rows = product(graph.values[x].shape[:-1])
    if not str(rows).isdigit():
        raise ValueError("topk_last requires literal extents so the grid is static")

    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr") for value in operands + results}
    pairs = [(ptr[value], Argument("tensor", value)) for value in operands + results]

    def constexpr(hint, literal):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(literal))))
        return symbol

    width_symbol = constexpr("WIDTH", width)
    k_symbol = constexpr("K", k)
    lanes = names.fresh("LANES")
    pairs.append((f"{lanes}: tl.constexpr", Argument("value", str(_pow2(width)))))

    row, cols, mask, tile = (names.fresh(hint) for hint in
                             ("row", "cols", "valid", "tile"))
    best, where, pick, step = (names.fresh(hint) for hint in
                               ("best", "hit", "pick", "step"))
    none = f"{best} == float(\"-inf\")"
    stores = []
    if keep_values:
        stores.append(f"        tl.store({ptr[values]} + {row} * {k_symbol} + {step},\n"
                      f"                 tl.where({none}, 0.0, {best}).to("
                      f"{_dtype_expr(graph, values)}))")
    if keep_indices:
        stores.append(f"        tl.store({ptr[indices]} + {row} * {k_symbol} + {step},"
                      f" {pick}.to(tl.int32))")
    source = f"""# topk_last: descending, ties to the lower column, NaN and -inf never selected
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
    {row} = tl.program_id(0)
    {cols} = tl.arange(0, {lanes})
    {mask} = {cols} < {width_symbol}
    {tile} = tl.load({ptr[x]} + {row} * {width_symbol} + {cols}, mask={mask},
                     other=float("-inf")).to(tl.float32)
    {tile} = tl.where({tile} == {tile}, {tile}, float("-inf"))
    for {step} in range({k_symbol}):
        {best} = tl.max({tile}, 0)
        {where} = {tile} == {best}
        {pick} = tl.where({none}, 0, tl.min(tl.where({where}, {cols}, {width_symbol}), 0))
{chr(10).join(stores)}
        {tile} = tl.where({cols} == {pick}, float("-inf"), {tile})
"""
    return TritonKernel(name, "topk", tuple(node.op for node in group.members),
                        operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs),
                        f"({rows},)", source,
                        launch_options=(("num_warps", config.num_warps),))


def gather_triton_kernel(graph, group, memory, name, config):
    """`out[..., j] = x[..., index[..., j]]`, one flat element per lane.

    Two stores rather than one: the zero region is written first and the gathered
    values second, so the load and the store that carries it are under the *same*
    mask. That is the shape the 昆仑 contract asks for ("store under exactly the
    load mask, or own the zero region with an allocation"), and it costs nothing
    because the operator is a gather anyway.
    """
    from flowgpu.compiler.backend.triton_backend import TritonKernel

    assert len(group.members) == 1 and group.anchor.op == "gather_last"
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 2 and len(results) == 1
    x, index = operands
    (out,) = results

    width = _literal(graph, x, -1, "gather_last")
    out_width = _literal(graph, out, -1, "gather_last")
    total = product(graph.values[out].shape)
    if not str(total).isdigit():
        raise ValueError("gather_last requires literal extents so the grid is static")
    total = int(total)

    names = Names(graph)
    ptr = {value: names.fresh(identifier(value) + "_ptr") for value in operands + results}
    pairs = [(ptr[value], Argument("tensor", value)) for value in operands + results]
    block = names.fresh("BLOCK")
    block_size = config.default_block_size
    pairs.append((f"{block}: tl.constexpr", Argument("value", str(block_size))))
    width_symbol = names.fresh("WIDTH")
    pairs.append((f"{width_symbol}: tl.constexpr", Argument("value", str(width))))
    out_width_symbol = names.fresh("OUT")
    pairs.append((f"{out_width_symbol}: tl.constexpr", Argument("value", str(out_width))))

    flat = names.fresh("flat")
    inside = names.fresh("inside")
    slot = names.fresh("slot")
    row = names.fresh("row")
    column = names.fresh("column")
    source_position = names.fresh("source")
    in_bounds = names.fresh("in_bounds")
    value = names.fresh("value")
    zero = "0.0" if graph.values[out].dtype in {"fp16", "bf16", "fp32"} else "0"
    # A 64-bit index must stay 64-bit: truncating it to int32 turns an
    # out-of-bounds value into a legal-looking one and reads the wrong element
    # (review P1-7, 2026-10-26).
    index_dtype = "tl.int64" if graph.values[index].dtype == "int64" else "tl.int32"
    source = f"""# gather_last: last-axis gather, negative index normalised once, out of bounds is zero
@triton.jit
def {name}({", ".join(parameter for parameter, _ in pairs)}):
    {flat} = tl.program_id(0) * {block} + tl.arange(0, {block})
    {inside} = {flat} < {total}
    {slot} = tl.load({ptr[index]} + {flat}, mask={inside}, other=0).to({index_dtype})
    {row} = {flat} // {out_width_symbol}
    {column} = {flat} - {row} * {out_width_symbol}
    {slot} = tl.where({slot} < 0, {slot} + {width_symbol}, {slot})
    {in_bounds} = {inside} & ({slot} >= 0) & ({slot} < {width_symbol})
    {source_position} = {row} * {width_symbol} + {slot}
    {value} = tl.load({ptr[x]} + {source_position}, mask={in_bounds}, other={zero})
    tl.store({ptr[out]} + {flat}, {zero}, mask={inside})
    tl.store({ptr[out]} + {flat}, {value}, mask={in_bounds})
"""
    chunks = (total + block_size - 1) // block_size
    return TritonKernel(name, "axis_gather", ("gather_last",), operands, results,
                        tuple(parameter for parameter, _ in pairs),
                        tuple(argument for _, argument in pairs),
                        f"({chunks},)", source,
                        launch_options=(("num_warps", config.num_warps),))


def topk_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.backends import storage_of
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    (x,) = operands
    values, indices = results
    # a result with no buffer is not stored; the kernel still produces the other
    keep_values = storage_of(memory, values) is not None
    keep_indices = storage_of(memory, indices) is not None
    # only the live results are kernel parameters: a dead result has no buffer
    # and the host cannot name one (review P1-6, 2026-10-26)
    results = tuple(value for value in results
                    if storage_of(memory, value) is not None)
    width = _literal(graph, x, -1, "topk_last")
    k = _literal(graph, values, -1, "topk_last")
    rows = product(graph.values[x].shape[:-1])
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    row, column, chosen, position = (names.fresh(hint) for hint in
                                     ("row", "column", "chosen", "position"))
    lines = [
        f"    for {row} in range({rows}):",
        f"        {chosen} = []",
        f"        for {column} in range({width}):",
        f"            value = float({variables[x]}[{row} * {width} + {column}])",
        # NaN and -inf are "no candidate": the same rule the emitted kernel states
        f"            if value == value and value != float('-inf'):",
        f"                {chosen}.append((value, {column}))",
        f"        {chosen}.sort(key=lambda entry: (-entry[0], entry[1]))",
    ]
    for slot in range(k):
        available = f"len({chosen}) > {slot}"
        lines.append(f"        {position} = {chosen}[{slot}] if {available} else (0.0, 0)")
        if keep_values:
            lines.append(f"        {variables[values]}[{row} * {k} + {slot}] = "
                         + _rounded(graph, values, f"{position}[0]"))
        if keep_indices:
            lines.append(f"        {variables[indices]}[{row} * {k} + {slot}] = {position}[1]")
    return _kernel(name, "topk", group, operands, results, pairs, lines)


def gather_reference_kernel(graph, group, memory, name):
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    x, index = operands
    (out,) = results
    width = _literal(graph, x, -1, "gather_last")
    out_width = _literal(graph, out, -1, "gather_last")
    total = product(graph.values[out].shape)
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs, [d for value in operands + results
                                        for d in graph.values[value].shape])
    flat, row, slot = (names.fresh(hint) for hint in ("flat", "row", "slot"))
    lines = [
        f"    for {flat} in range({total}):",
        f"        {row} = {flat} // {out_width}",
        f"        {slot} = {variables[index]}[{flat}]",
        f"        if {slot} < 0: {slot} += {width}",
        f"        if 0 <= {slot} < {width}:",
        f"            {variables[out]}[{flat}] = " + _rounded(
            graph, out, f"{variables[x]}[{row} * {width} + {slot}]"),
        f"        else:",
        f"            {variables[out]}[{flat}] = " + _rounded(graph, out, "0"),
    ]
    return _kernel(name, "axis_gather", group, operands, results, pairs, lines)
