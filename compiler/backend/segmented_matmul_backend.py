"""Dedicated one-launch lowering for `segmented_matmul`: a grouped GEMM.

    out[t, :] = x[t, :] @ w[segment(t), :, :]

`indptr` is an explicit int32 tensor naming the segment boundaries, so the token
axis stays *symbolic* (no dimension arithmetic is performed on it) and the weight
index is the segment id rather than a computed address. That is what keeps this a
declarative operator rather than general indexing, and it is the `segment`
capability the catalogue survey found blocking thirteen tasks.

Two emission forms, chosen by whether the reduction (`K`) and column (`N`)
extents are literals. Both walk one segment's rows with a dynamic loop, but the
product they compute with has to exist as a compile-time tile:

* **literal K and N** -- the original form. `tl.arange(0, K)` over the whole
  reduction, one `tl.sum` product per row, one store. This is the byte form the
  2026-10-20 H100 run validated and the one `examples/batch10/step_a_v.fgpu`
  used while `segmented_matmul` required literal extents.

* **symbolic K or N** -- the shape-generic form added 2026-10-21 for the Task157
  submission. A compiled module is tied to whatever extents its DSL declaration
  spells out, so a program written `bf16[3,16,8]` can only ever serve KV=16,
  rank=8. The official harness chooses its own shapes, and a wrong answer is 0/7
  on every chip; so the symbolic form reads `K`, `N` and the row-tile count from
  the host at launch (they are plain arguments, exactly like a declared scalar)
  and walks them in `BLOCK_K`/`BLOCK_N` chunks. The row axis is also tiled
  (`BLOCK_M` rows per program) so the grid is `segments x row-tiles`: without it
  one program per segment is serial across every (token, head) row, which is
  enough parallelism for a toy shape and not for a benchmark.

  The symbolic form deliberately loads with *clamped addresses plus
  `tl.where`* and a masked store, never a masked load. The 2026-10-19 Task144
  evidence is that 昆仑芯 mismatched on every attempt that used a masked load
  (1e38 = uninitialized memory) and passed 7/7 on the attempt that removed them
  in favour of address clamping; we cannot test that chip locally, so the
  emission follows the only form that has passed on it. For a shape whose K and
  N are multiples of the block sizes the masks are identically true anyway.
"""

from compiler.backend.backends import (
    Argument,
    Names,
    group_operands,
    group_results,
    identifier,
)
from compiler.core.types import is_symbol


def _pow2(value):
    """The next power of two >= value (Triton's tl.arange requires one)."""
    width = 1
    while width < value:
        width *= 2
    return width


def _literal(graph, value, axis, what):
    extent = str(graph.values[value].shape[axis])
    if not extent.isdigit():
        raise ValueError(f"segmented_matmul requires a literal {what} extent")
    return int(extent)


def _extent(graph, value, axis):
    """The declared extent of one axis, as written: a digit string or a symbol."""
    return str(graph.values[value].shape[axis])


# Generic-form tile sizes. They are *not* tuning constants: they are the
# smallest powers of two that keep the per-program register tile small. The
# row tile is what makes the grid `segments x cdiv(rows, BLOCK_M)`; BLOCK_K and
# BLOCK_N are the loop steps over the runtime reduction/column extents, so any
# K or N is covered and only the last chunk of each is masked.
GENERIC_BLOCK_M = 32
GENERIC_BLOCK_K = 128
GENERIC_BLOCK_N = 64


def _literal_kernel(graph, group, memory, name, config):
    """The original emission: literal `K`/`N` constexprs, whole-row product."""
    from compiler.backend.triton_backend import TritonKernel

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 3 and len(results) == 1
    x, w, indptr = operands
    out = results[0]
    reduction = _literal(graph, x, 1, "reduction")
    columns = _literal(graph, w, 2, "column")
    names = Names(graph)
    x_ptr = names.fresh(identifier(x) + "_ptr")
    w_ptr = names.fresh(identifier(w) + "_ptr")
    i_ptr = names.fresh(identifier(indptr) + "_ptr")
    out_ptr = names.fresh(identifier(out) + "_ptr")
    pairs = [(x_ptr, Argument("tensor", x)), (w_ptr, Argument("tensor", w)),
             (i_ptr, Argument("tensor", indptr)), (out_ptr, Argument("tensor", out))]

    def constexpr(hint, argument):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", argument))
        return symbol

    k_symbol = constexpr("K", Argument("value", str(reduction)))
    n_symbol = constexpr("N", Argument("value", str(columns)))
    block_n = names.fresh("BLOCK_N")
    block_k = names.fresh("BLOCK_K")
    # tl.arange requires a power of two, and a literal K/N need not be one (the
    # test shape 4x3 is not).  The lanes are widened to the next power of two and
    # masked, and N is walked in column blocks: the old form silently dropped
    # every column past the first block (review P0-3/P1-9, 2026-10-26).
    columns_per_block = _pow2(min(columns, config.default_block_size))
    reduction_width = _pow2(reduction)
    pairs.append((f"{block_n}: tl.constexpr", Argument("value", str(columns_per_block))))
    pairs.append((f"{block_k}: tl.constexpr", Argument("value", str(reduction_width))))

    segment = names.fresh("segment")
    first = names.fresh("first")
    last = names.fresh("last")
    row = names.fresh("row")
    lanes = names.fresh("lanes")
    keep = names.fresh("keep")
    k_lanes = names.fresh("k_lanes")
    k_keep = names.fresh("k_keep")
    col_block = names.fresh("col_block")
    cols = names.fresh("cols")
    acc = names.fresh("acc")
    x_row = names.fresh("x_row")
    w_tile = names.fresh("w_tile")
    lines = [
        f"    {segment} = tl.program_id(0)",
        f"    {first} = tl.load({i_ptr} + {segment}).to(tl.int32)",
        f"    {last} = tl.load({i_ptr} + {segment} + 1).to(tl.int32)",
        f"    {lanes} = tl.arange(0, {block_n})",
        f"    {k_lanes} = tl.arange(0, {block_k})",
        f"    {k_keep} = {k_lanes} < {k_symbol}",
        f"    for {row} in range({first}, {last}):",
        f"        {x_row} = tl.load({x_ptr} + {row} * {k_symbol} + {k_lanes},"
        f" mask={k_keep}, other=0.0).to(tl.float32)",
        f"        for {col_block} in range(0, tl.cdiv({n_symbol}, {block_n})):",
        f"            {cols} = {col_block} * {block_n} + {lanes}",
        f"            {keep} = {cols} < {n_symbol}",
        f"            {w_tile} = tl.load({w_ptr} + {segment} * ({k_symbol} * {n_symbol})"
        f" + {k_lanes}[:, None] * {n_symbol} + {cols}[None, :],"
        f" mask={k_keep}[:, None] & {keep}[None, :], other=0.0).to(tl.float32)",
        f"            {acc} = tl.sum({x_row}[:, None] * {w_tile}, axis=0)",
        f"            tl.store({out_ptr} + {row} * {n_symbol} + {cols},"
        f" {acc}.to({out_ptr}.dtype.element_ty), mask={keep})",
    ]
    parameters = tuple(parameter for parameter, _ in pairs)
    source = "# segmented_matmul\n@triton.jit\n"
    source += f"def {name}({', '.join(parameters)}):\n" + "\n".join(lines) + "\n"
    return TritonKernel(name, "segmented_matmul", tuple(node.op for node in group.members),
                        operands, results, parameters,
                        tuple(argument for _, argument in pairs),
                        # The grid is a *host* expression: it reads the tensor
                        # argument, not the kernel's pointer parameter. Using
                        # `w_ptr` here compiled fine and failed at launch with
                        # NameError (found on the H100, 2026-10-20).
                        f"({identifier(w)}.shape[0],)", source,
                        launch_options=(("num_warps", config.num_warps),))


def _generic_kernel(graph, group, memory, name, config):
    """The shape-generic emission: runtime K/N extents, row-tiled 1-D grid.

    Every extent that is not a literal reaches the kernel as a plain positional
    argument. The host already binds symbolic dimensions from the input tensors
    (``T = x_flat.shape[0]``) and can pass a derived expression for the tile
    count, so no new host mechanism is involved -- the extent is passed the way
    a declared scalar is.
    """
    from compiler.backend.triton_backend import TritonKernel

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 3 and len(results) == 1
    x, w, indptr = operands
    out = results[0]
    rows_extent = _extent(graph, x, 0)
    reduction_extent = _extent(graph, x, 1)
    columns_extent = _extent(graph, w, 2)
    names = Names(graph)
    x_ptr = names.fresh(identifier(x) + "_ptr")
    w_ptr = names.fresh(identifier(w) + "_ptr")
    i_ptr = names.fresh(identifier(indptr) + "_ptr")
    out_ptr = names.fresh(identifier(out) + "_ptr")
    pairs = [(x_ptr, Argument("tensor", x)), (w_ptr, Argument("tensor", w)),
             (i_ptr, Argument("tensor", indptr)), (out_ptr, Argument("tensor", out))]

    def runtime(hint, expression):
        """A plain (non-constexpr) formal, passed the given host expression."""
        symbol = names.fresh(hint)
        pairs.append((symbol, Argument("value", expression)))
        return symbol

    def constexpr(hint, literal):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(literal))))
        return symbol

    k_symbol = runtime("K", reduction_extent)
    n_symbol = runtime("N", columns_extent)
    block_m = constexpr("BLOCK_M", GENERIC_BLOCK_M)
    block_k = constexpr("BLOCK_K", GENERIC_BLOCK_K)
    block_n = constexpr("BLOCK_N", GENERIC_BLOCK_N)
    # Segment `pid // tiles` and row tile `pid % tiles`: the grid is the
    # product, so both are recovered from one dimension on every backend that
    # accepts a 1-D grid (the form this kernel already launches with).
    tiles = runtime("tiles", f"triton.cdiv({rows_extent}, {GENERIC_BLOCK_M})")

    segment = names.fresh("segment")
    tile = names.fresh("tile")
    first = names.fresh("first")
    last = names.fresh("last")
    begin = names.fresh("begin")
    end = names.fresh("end")
    row = names.fresh("row")
    column0 = names.fresh("column0")
    columns = names.fresh("columns")
    keep = names.fresh("keep")
    column_last = names.fresh("column_last")
    acc = names.fresh("acc")
    k0 = names.fresh("k0")
    k_lanes = names.fresh("k_lanes")
    k_offsets = names.fresh("k_offsets")
    k_last = names.fresh("k_last")
    k_in = names.fresh("k_in")
    x_row = names.fresh("x_row")
    x_masked = names.fresh("x_masked")
    w_tile = names.fresh("w_tile")
    w_masked = names.fresh("w_masked")
    store_columns = names.fresh("store_columns")
    lines = [
        f"    {segment} = tl.program_id(0) // {tiles}",
        f"    {tile} = tl.program_id(0) % {tiles}",
        f"    {first} = tl.load({i_ptr} + {segment}).to(tl.int32)",
        f"    {last} = tl.load({i_ptr} + {segment} + 1).to(tl.int32)",
        f"    {begin} = tl.maximum({first}, {tile} * {block_m})",
        f"    {end} = tl.minimum({last}, ({tile} + 1) * {block_m})",
        f"    {columns} = tl.arange(0, {block_n})",
        f"    {k_lanes} = tl.arange(0, {block_k})",
        f"    for {row} in range({begin}, {end}):",
        f"        for {column0} in range(0, {n_symbol}, {block_n}):",
        f"            {column_last} = {n_symbol} - 1",
        f"            {store_columns} = {column0} + {columns}",
        f"            {keep} = {store_columns} < {n_symbol}",
        f"            {acc} = tl.zeros(({block_n},), tl.float32)",
        f"            for {k0} in range(0, {k_symbol}, {block_k}):",
        # Clamped addresses + `tl.where` instead of masked loads: 昆仑芯 has
        # never passed a masked-load form locally (Task144 evidence) and this
        # form is the one it did pass. Both clamps are in-bounds by construction.
        f"                {k_in} = {k0} + {k_lanes}",
        f"                {k_last} = {k_symbol} - 1",
        f"                {k_offsets} = tl.minimum({k_in}, {k_last})",
        f"                {x_row} = tl.load({x_ptr} + {row} * {k_symbol} + {k_offsets}).to(tl.float32)",
        f"                {x_masked} = tl.where({k_in} < {k_symbol}, {x_row}, 0.0)",
        f"                {w_tile} = tl.load({w_ptr} + {segment} * ({k_symbol} * {n_symbol})"
        f" + {k_offsets}[:, None] * {n_symbol}"
        f" + tl.minimum({store_columns}, {column_last})[None, :]).to(tl.float32)",
        f"                {w_masked} = tl.where(({k_in} < {k_symbol})[:, None] & {keep}[None, :],"
        f" {w_tile}, 0.0)",
        f"                {acc} = {acc} + tl.sum({x_masked}[:, None] * {w_masked}, axis=0)",
        f"            tl.store({out_ptr} + {row} * {n_symbol}"
        f" + tl.minimum({store_columns}, {column_last}),"
        f" {acc}.to({out_ptr}.dtype.element_ty), mask={keep})",
    ]
    parameters = tuple(parameter for parameter, _ in pairs)
    source = "# segmented_matmul (shape-generic)\n@triton.jit\n"
    source += f"def {name}({', '.join(parameters)}):\n" + "\n".join(lines) + "\n"
    grid = (f"({identifier(w)}.shape[0]"
            f" * triton.cdiv({rows_extent}, {GENERIC_BLOCK_M}),)")
    return TritonKernel(name, "segmented_matmul", tuple(node.op for node in group.members),
                        operands, results, parameters,
                        tuple(argument for _, argument in pairs),
                        grid, source,
                        launch_options=(("num_warps", config.num_warps),))


def triton_kernel(graph, group, memory, name, config):
    """Dispatch on the operator, then on the extents / dtype.

    `segmented_lora_matmul` selects its weight per segment from an adapter
    stack, so its emission also depends on the element dtype (the tensor-core
    dot form is only exact for the 16-bit floats). `segmented_matmul` keeps two
    forms chosen by whether the reduction and column extents are literals:
    literal extents keep the original bytes, so every program already emitted
    that way (and the H100 run that validated it) is unchanged. Genericity is
    opt-in by writing the DSL declaration with symbols, which is what a module
    that must meet a harness-chosen shape has to do.
    """
    assert len(group.members) == 1
    anchor = group.anchor.op
    if anchor == "segmented_lora_matmul":
        operands = group_operands(graph, group)
        if graph.values[operands[0]].dtype in LORA_DOT_DTYPES:
            return _lora_dot_kernel(graph, group, memory, name, config)
        return _lora_sum_kernel(graph, group, memory, name, config)
    assert anchor == "segmented_matmul"
    assert all(node.op == "segmented_matmul" for node in group.members)
    operands = group_operands(graph, group)
    x, w, _ = operands
    literal = (_extent(graph, x, 1).isdigit()
               and _extent(graph, w, 2).isdigit()
               and not is_symbol(_extent(graph, x, 0)))
    if literal:
        return _literal_kernel(graph, group, memory, name, config)
    return _generic_kernel(graph, group, memory, name, config)


def reference_kernel(graph, group, memory, name):
    """Host-side reference: the definition, written out segment by segment."""
    if group.anchor.op == "segmented_lora_matmul":
        return _lora_reference_kernel(graph, group, memory, name)
    from compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    x, w, indptr = operands
    out = results[0]
    reduction = _extent(graph, x, 1)
    columns = _extent(graph, w, 2)
    reduction_text = reduction if is_symbol(reduction) else reduction
    columns_text = columns if is_symbol(columns) else columns
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + results]
    _dimension_parameters(names, pairs,
                          [d for v in operands + results for d in graph.values[v].shape])
    lines = [
        f"    for segment in range(len({variables[indptr]}) - 1):",
        f"        first = {variables[indptr]}[segment]",
        f"        last = {variables[indptr]}[segment + 1]",
        f"        for row in range(first, last):",
        f"            for column in range({columns_text}):",
        f"                total = 0.0",
        f"                for k in range({reduction_text}):",
        f"                    total = total + {variables[x]}[row * {reduction} + k]"
        f" * {variables[w]}[(segment * {reduction} + k) * {columns} + column]",
        f"                {variables[out]}[row * {columns} + column] = "
        f"{_rounded(graph, out, 'total')}",
    ]
    return _kernel(name, "segmented_matmul", group, operands, results, pairs, lines)


# ---------------------------------------------------------------------------
# `segmented_lora_matmul`: the grouped GEMM whose weight is *selected* per
# segment from a shared adapter stack. See the type rule in
# compiler/frontend/operator.py for the contract.
#
# Two emissions, chosen by the element dtype:
#
# * bf16/fp16 -> a `tl.dot` tile. bf16 and fp16 widen to fp32 exactly (their
#   significands, 8 and 11 bits, multiply into <= 22 bits, well inside fp32's
#   24), so a tensor-core dot accumulating in fp32 computes the same products
#   the reference's `x.float() @ a.float().T` does, up to fp32 reduction order.
#   This is the form every Triton kernel on this task family uses, including
#   the task's own SGLang baseline, so it is the form the foreign backends have
#   to support anyway; the row-serial form below is a category slower.
#
# * fp32 -> the row-serial `tl.sum` form. `tl.dot` on fp32 inputs is IEEE only
#   with a backend-specific precision argument (`input_precision='ieee'` in
#   Triton 3.x, which is not in every vendor fork's API); without it the CUDA
#   path uses TF32 and the fp32 tolerance (1e-4) rejects the result. fp32 is
#   not the task's stated dtype, so the exact-but-slow form is the honest
#   fallback rather than a precision gamble.
LORA_DOT_DTYPES = ("bf16", "fp16")
LORA_BLOCK_M = 32
LORA_BLOCK_K = 128
LORA_BLOCK_N = 64


def _lora_operands(graph, group):
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    assert len(operands) == 6 and len(results) == 1
    return operands, results


def _lora_pairs(graph, names, operands, results):
    """The six operands as kernel arguments: five pointers and `heads`.

    The scalar is a plain Triton argument (`Argument('scalar', ...)`), the same
    contract a declared scalar reaches a pointwise kernel with: a number, never
    a pointer. The adapter passes the head count as a Python int.
    """
    pointers = [(names.fresh(identifier(value) + "_ptr"), Argument("tensor", value))
                for value in operands[:5]]
    pointers.append((names.fresh(identifier(results[0]) + "_ptr"),
                     Argument("tensor", results[0])))
    scalar = names.fresh(identifier(operands[5]))
    pairs = list(pointers)
    pairs.append((scalar, Argument("scalar", operands[5])))
    return pairs, scalar


def _lora_dot_kernel(graph, group, memory, name, config):
    from compiler.backend.triton_backend import TritonKernel

    operands, results = _lora_operands(graph, group)
    x, a, indptr, weight_indices, lora_ranks, heads = operands
    out = results[0]
    reduction = _extent(graph, a, 2)
    columns = _extent(graph, a, 1)
    rows = _extent(graph, x, 0)
    names = Names(graph)
    pairs, heads_symbol = _lora_pairs(graph, names, operands, results)

    def runtime(hint, expression):
        symbol = names.fresh(hint)
        pairs.append((symbol, Argument("value", expression)))
        return symbol

    def constexpr(hint, literal):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(literal))))
        return symbol

    k_symbol = runtime("K", reduction)
    n_symbol = runtime("N", columns)
    block_m = constexpr("BLOCK_M", LORA_BLOCK_M)
    block_k = constexpr("BLOCK_K", LORA_BLOCK_K)
    block_n = constexpr("BLOCK_N", LORA_BLOCK_N)

    x_ptr, a_ptr, i_ptr, wi_ptr, lr_ptr, out_ptr = (parameter for parameter, _ in pairs[:6])
    segment = names.fresh("segment")
    tile = names.fresh("tile")
    weight_index = names.fresh("weight_index")
    adapter_rank = names.fresh("adapter_rank")
    first = names.fresh("first")
    last = names.fresh("last")
    rows_symbol = names.fresh("rows")
    row_in = names.fresh("row_in")
    row_safe = names.fresh("row_safe")
    cols = names.fresh("cols")
    col0 = names.fresh("column0")
    col_in = names.fresh("col_in")
    col_safe = names.fresh("col_safe")
    k_lanes = names.fresh("k_lanes")
    acc = names.fresh("acc")
    k0 = names.fresh("k0")
    k_in = names.fresh("k_in")
    k_safe = names.fresh("k_safe")
    x_tile = names.fresh("x_tile")
    w_tile = names.fresh("w_tile")
    live = names.fresh("live")
    lines = [
        f"    {segment} = tl.program_id(0)",
        f"    {tile} = tl.program_id(1)",
        f"    {weight_index} = tl.load({wi_ptr} + {segment}).to(tl.int32)",
        f"    {adapter_rank} = tl.load({lr_ptr} + {weight_index}).to(tl.int32)",
        # indptr is in token space; the adapter passes the head count so the
        # kernel flattens (token, head) itself -- no per-call host kernel.
        f"    {first} = tl.load({i_ptr} + {segment}).to(tl.int32) * {heads_symbol}",
        f"    {last} = tl.load({i_ptr} + {segment} + 1).to(tl.int32) * {heads_symbol}",
        f"    {live} = {adapter_rank} > 0",
        f"    {rows_symbol} = {first} + {tile} * {block_m} + tl.arange(0, {block_m})",
        f"    {row_in} = {rows_symbol} < {last}",
        f"    {row_safe} = tl.maximum(tl.minimum({rows_symbol}, {last} - 1), 0)",
        f"    {cols} = tl.arange(0, {block_n})",
        f"    {k_lanes} = tl.arange(0, {block_k})",
        f"    for {col0} in range(0, {n_symbol}, {block_n}):",
        f"        {col_in} = {col0} + {cols} < {n_symbol}",
        f"        {col_safe} = tl.minimum({col0} + {cols}, {n_symbol} - 1)",
        f"        {acc} = tl.zeros(({block_m}, {block_n}), tl.float32)",
        f"        for {k0} in range(0, {k_symbol}, {block_k}):",
        f"            {k_in} = {k0} + {k_lanes} < {k_symbol}",
        f"            {k_safe} = tl.minimum({k0} + {k_lanes}, {k_symbol} - 1)",
        # Clamped addresses + `tl.where` rather than masked loads: 昆仑芯 has
        # never passed a masked-load kernel locally (Task144 evidence) and the
        # where form is the one it did pass. The clamp keeps every address in
        # range, so no predicate is needed for safety, only for correctness.
        f"            {x_tile} = tl.load({x_ptr} + {row_safe}[:, None] * {k_symbol}"
        f" + {k_safe}[None, :])",
        f"            {x_tile} = tl.where({row_in}[:, None] & {k_in}[None, :], {x_tile},"
        f" tl.zeros(({block_m}, {block_k}), {x_ptr}.dtype.element_ty))",
        f"            {w_tile} = tl.load({a_ptr} + {weight_index} * ({n_symbol} * {k_symbol})"
        f" + {col_safe}[None, :] * {k_symbol} + {k_safe}[:, None])",
        f"            {w_tile} = tl.where({k_in}[:, None] & {col_in}[None, :], {w_tile},"
        f" tl.zeros(({block_k}, {block_n}), {a_ptr}.dtype.element_ty))",
        f"            {acc} = tl.dot({x_tile}, {w_tile}, {acc})",
        f"        {acc} = tl.where({live} & {row_in}[:, None] & {col_in}[None, :], {acc}, 0.0)",
        f"        tl.store({out_ptr} + {row_safe}[:, None] * {n_symbol} + {col_safe}[None, :],"
        f" {acc}.to({out_ptr}.dtype.element_ty), mask={row_in}[:, None] & {col_in}[None, :])",
    ]
    parameters = tuple(parameter for parameter, _ in pairs)
    source = "# segmented_lora_matmul (tiled dot)\n@triton.jit\n"
    source += f"def {name}({', '.join(parameters)}):\n" + "\n".join(lines) + "\n"
    grid = (f"({identifier(weight_indices)}.shape[0],"
            f" triton.cdiv({rows}, {LORA_BLOCK_M}))")
    return TritonKernel(name, "segmented_matmul", tuple(node.op for node in group.members),
                        operands, results, parameters,
                        tuple(argument for _, argument in pairs), grid, source,
                        launch_options=(("num_warps", config.num_warps),))


def _lora_sum_kernel(graph, group, memory, name, config):
    """The exact fp32 form: per row, one `tl.sum` product over the reduction."""
    from compiler.backend.triton_backend import TritonKernel

    operands, results = _lora_operands(graph, group)
    x, a, indptr, weight_indices, lora_ranks, heads = operands
    out = results[0]
    reduction = _extent(graph, a, 2)
    columns = _extent(graph, a, 1)
    rows = _extent(graph, x, 0)
    names = Names(graph)
    pairs, heads_symbol = _lora_pairs(graph, names, operands, results)

    def runtime(hint, expression):
        symbol = names.fresh(hint)
        pairs.append((symbol, Argument("value", expression)))
        return symbol

    def constexpr(hint, literal):
        symbol = names.fresh(hint)
        pairs.append((f"{symbol}: tl.constexpr", Argument("value", str(literal))))
        return symbol

    k_symbol = runtime("K", reduction)
    n_symbol = runtime("N", columns)
    block_m = constexpr("BLOCK_M", LORA_BLOCK_M)
    block_k = constexpr("BLOCK_K", LORA_BLOCK_K)
    block_n = constexpr("BLOCK_N", LORA_BLOCK_N)

    x_ptr, a_ptr, i_ptr, wi_ptr, lr_ptr, out_ptr = (parameter for parameter, _ in pairs[:6])
    segment = names.fresh("segment")
    tile = names.fresh("tile")
    weight_index = names.fresh("weight_index")
    adapter_rank = names.fresh("adapter_rank")
    first = names.fresh("first")
    last = names.fresh("last")
    begin = names.fresh("begin")
    end = names.fresh("end")
    row = names.fresh("row")
    column0 = names.fresh("column0")
    cols = names.fresh("cols")
    keep = names.fresh("keep")
    col_safe = names.fresh("col_safe")
    acc = names.fresh("acc")
    k_lanes = names.fresh("k_lanes")
    k0 = names.fresh("k0")
    k_in = names.fresh("k_in")
    k_safe = names.fresh("k_safe")
    x_row = names.fresh("x_row")
    w_tile = names.fresh("w_tile")
    live = names.fresh("live")
    lines = [
        f"    {segment} = tl.program_id(0)",
        f"    {tile} = tl.program_id(1)",
        f"    {weight_index} = tl.load({wi_ptr} + {segment}).to(tl.int32)",
        f"    {adapter_rank} = tl.load({lr_ptr} + {weight_index}).to(tl.int32)",
        f"    {first} = tl.load({i_ptr} + {segment}).to(tl.int32) * {heads_symbol}",
        f"    {last} = tl.load({i_ptr} + {segment} + 1).to(tl.int32) * {heads_symbol}",
        f"    {begin} = {first} + {tile} * {block_m}",
        f"    {end} = tl.minimum({last}, {begin} + {block_m})",
        f"    {cols} = tl.arange(0, {block_n})",
        f"    {k_lanes} = tl.arange(0, {block_k})",
        f"    {live} = {adapter_rank} > 0",
        f"    for {row} in range({begin}, {end}):",
        f"        for {column0} in range(0, {n_symbol}, {block_n}):",
        f"            {keep} = {column0} + {cols} < {n_symbol}",
        f"            {col_safe} = tl.minimum({column0} + {cols}, {n_symbol} - 1)",
        f"            {acc} = tl.zeros(({block_n},), tl.float32)",
        f"            for {k0} in range(0, {k_symbol}, {block_k}):",
        f"                {k_in} = {k0} + {k_lanes} < {k_symbol}",
        f"                {k_safe} = tl.minimum({k0} + {k_lanes}, {k_symbol} - 1)",
        f"                {x_row} = tl.load({x_ptr} + {row} * {k_symbol} + {k_safe}).to(tl.float32)",
        f"                {x_row} = tl.where({k_in}, {x_row}, 0.0)",
        f"                {w_tile} = tl.load({a_ptr} + {weight_index}"
        f" * ({n_symbol} * {k_symbol}) + {col_safe}[None, :] * {k_symbol}"
        f" + {k_safe}[:, None]).to(tl.float32)",
        f"                {w_tile} = tl.where({k_in}[:, None] & {keep}[None, :], {w_tile}, 0.0)",
        f"                {acc} = {acc} + tl.sum({x_row}[:, None] * {w_tile}, axis=0)",
        f"            {acc} = tl.where({live} & {keep}, {acc}, 0.0)",
        f"            tl.store({out_ptr} + {row} * {n_symbol} + {col_safe},"
        f" {acc}.to({out_ptr}.dtype.element_ty), mask={keep})",
    ]
    parameters = tuple(parameter for parameter, _ in pairs)
    source = "# segmented_lora_matmul (row-serial, exact fp32)\n@triton.jit\n"
    source += f"def {name}({', '.join(parameters)}):\n" + "\n".join(lines) + "\n"
    grid = (f"({identifier(weight_indices)}.shape[0],"
            f" triton.cdiv({rows}, {LORA_BLOCK_M}))")
    return TritonKernel(name, "segmented_matmul", tuple(node.op for node in group.members),
                        operands, results, parameters,
                        tuple(argument for _, argument in pairs), grid, source,
                        launch_options=(("num_warps", config.num_warps),))


def _lora_reference_kernel(graph, group, memory, name):
    from compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands, results = _lora_operands(graph, group)
    x, a, indptr, weight_indices, lora_ranks, heads = operands
    out = results[0]
    reduction = _extent(graph, a, 2)
    columns = _extent(graph, a, 1)
    names = Names(graph)
    tensor_operands = operands[:3] + (weight_indices, lora_ranks)
    variables = {value: names.reserve(identifier(value))
                 for value in tensor_operands + results}
    heads_name = names.reserve(identifier(heads))
    pairs = [(variables[value], Argument("tensor", value)) for value in tensor_operands]
    pairs += [(variables[out], Argument("tensor", out))]
    pairs += [(heads_name, Argument("scalar", heads))]
    _dimension_parameters(names, pairs, [d for v in tensor_operands + results
                                        for d in graph.values[v].shape])
    # A rank-0 adapter is skipped by the task's reference, which allocates the
    # destination with zeros and never writes the segment. This lowering's
    # destination starts uninitialized (that is what lets a fused producer write
    # it), so the skip *writes* the zeros instead of leaving them to the caller:
    # the emitted kernel does the same, and the two agree on every row.
    lines = [
        f"    for segment in range(len({variables[indptr]}) - 1):",
        f"        first = {variables[indptr]}[segment] * {heads_name}",
        f"        last = {variables[indptr]}[segment + 1] * {heads_name}",
        f"        weight_index = {variables[weight_indices]}[segment]",
        f"        keep = {variables[lora_ranks]}[weight_index] != 0",
        f"        for row in range(first, last):",
        f"            for column in range({columns}):",
        f"                total = 0.0",
        f"                if keep:",
        f"                    for k in range({reduction}):",
        f"                        total = total + {variables[x]}[row * {reduction} + k]"
        f" * {variables[a]}[(weight_index * {columns} + column) * {reduction} + k]",
        f"                {variables[out]}[row * {columns} + column] = "
        f"{_rounded(graph, out, 'total')}",
    ]
    return _kernel(name, "segmented_matmul", group, operands, results, pairs, lines)
