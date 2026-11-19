"""Lowering for `store_at`: a contiguous block write at a device-table base.

The language contract is in ``docs/STORE_AT_DESIGN_20261022.md`` and the staged
type rule in ``compiler/frontend/store_at.py``. This module emits the smallest
kernel that can express it:

    p    = tl.program_id(0)
    base = tl.load(offsets_ptr + p).to(tl.int64)   # program-uniform, once
    lane = tl.arange(0, BLOCK)
    mask = lane < LANE
    v    = tl.load(values_ptr + p * LANE + lane, mask=mask, other=0)
    tl.store(out_ptr + base + lane, v, mask=mask)

Every quantity except `offsets[p]` is affine in the lane, so the write address is
`dst_ptr + scalar_base + affine(arange)` -- the only write shape the measured
targets sustain. The emitter proves that itself on its own output through
``analysis/write_address`` (contract 3): a store_at lowering whose store address
is not affine refuses to emit, independently of any platform policy.

Functional semantics: the result is a fresh value equal to `dst` with the runs
replaced. When the runs provably tile `dst` (all shapes literal and
`dst.numel() == rows * lane_count`) the copy is dead and only the write kernel is
emitted -- one launch, one pass. Otherwise a copy kernel is emitted first, and
it is launched only when the *host-side shape comparison* says the runs do not
cover the destination; that branch is pure metadata and never reads tensor data.
"""
from dataclasses import dataclass, replace

from compiler.backend.backends import (
    Argument,
    Names,
    identifier,
    product,
    group_operands,
    group_results,
    storage_of,
)
from compiler.core.diagnostic import CompileError, DiagnosticKind
from compiler.ir.dataflow import resolve_alias


@dataclass(frozen=True)
class StoreAtContract:
    """What the emitted host has to enforce for one `store_at` call.

    Recorded on the write kernel so ``_host`` can emit the fail-closed checks
    before any allocation or launch. `lane_count` and `dst_elements` are host
    expressions over declared shape symbols, never tensor data: the coverage
    decision (`dst_elements != rows * lane_count`) is therefore pure metadata,
    while the bounds/overlap check deliberately reads the offsets values
    (one D2H + sync) unless the author used the trusted spelling.
    """

    kernel: str
    dst: str
    offsets: str
    values: str
    lane_count: str
    dst_elements: str
    coverage: str | None
    trusted: bool = False


def _literal_product(dimensions):
    """The product of literal dimensions, or None when one is symbolic."""
    total = 1
    for dimension in dimensions:
        if not str(dimension).isdigit():
            return None
        total *= int(dimension)
    return total


def triton_kernel(graph, group, memory, name, config):
    from compiler.backend.triton_backend import TritonKernel
    from compiler.backend.program_scalar import emit_uniform_load
    from compiler.ir.program_scalar_ir import ProgramScalar, ProgramIndex

    assert len(group.members) == 1
    node = group.members[0]
    trusted = node.op.endswith("_trusted")
    operands = group_operands(graph, group)
    results = group_results(graph, group)
    dtype = graph.values[results[0]].dtype
    assert len(operands) == 3 and len(results) == 1
    dst, offsets, values = operands
    out = results[0]

    # The fail-closed bounds/overlap check reads the offsets values on the host
    # before launching. That is only possible when the table is a caller-provided
    # tensor; a device-produced table would need its producer's launch first, so
    # that case is refused rather than silently checked later (it is the next
    # lowering step, not a hole).
    offsets_storage = storage_of(memory, resolve_alias(graph.aliases, offsets))
    if offsets_storage != "input":
        raise CompileError(
            DiagnosticKind.OPERATOR,
            f"store_at offsets '{offsets}' must be a caller-provided input "
            f"tensor: the fail-closed bounds/overlap contract reads their values "
            f"on the host before any allocation or launch. A device-produced "
            f"offsets table needs the next lowering step",
            pipeline=graph.pipeline,
        )

    values_shape = graph.values[values].shape
    lane_count = product(values_shape[1:])
    lane_literal = _literal_product(values_shape[1:])
    dst_shape = graph.values[dst].shape
    dst_elements = product(dst_shape)
    dst_literal = _literal_product(dst_shape)
    rows = str(values_shape[0])
    rows_literal = int(rows) if rows.isdigit() else None

    # Coverage is a shape fact and stays one: no tensor data is consulted. When
    # both sides are literal, equality is decided now and the copy kernel is not
    # emitted at all; otherwise the copy is emitted with a metadata-only guard.
    static_coverage = (lane_literal is not None and dst_literal is not None
                       and rows_literal is not None
                       and dst_literal == rows_literal * lane_literal)
    coverage_guard = None
    if not static_coverage:
        if lane_literal is not None and dst_literal is not None and rows_literal is not None:
            coverage_guard = None      # provably unequal: the copy is always needed
        else:
            coverage_guard = (f"({dst_elements}) != ({rows}) * ({lane_count})")

    names = Names(graph)
    pointers = {}
    for value in (offsets, values, out):
        pointers[value] = names.fresh(identifier(value) + "_ptr")
    write_pairs = [(pointers[value], Argument("tensor", value))
                   for value in (offsets, values, out)]
    lane_symbol = names.fresh("LANE")
    block_symbol = names.fresh("BLOCK")
    write_pairs.append((f"{lane_symbol}: tl.constexpr",
                        Argument("value", lane_count)))
    write_pairs.append((f"{block_symbol}: tl.constexpr",
                        Argument("value", f"triton.next_power_of_2({lane_count})")))

    program = names.fresh("p")
    lane = names.fresh("lane")
    mask = names.fresh("mask")
    value = names.fresh("value")
    lines = [f"    {program} = tl.program_id(0)"]
    spec = ProgramScalar(table=offsets, index=ProgramIndex(axis=0),
                         dtype="int64", bounds="trusted")
    base, _bound = emit_uniform_load(
        spec, offsets, "base", graph, names, write_pairs, {},
        lines, grid_count=rows, index_source_override=program,
    )
    lines += [
        f"    {lane} = tl.arange(0, {block_symbol})",
        f"    {mask} = {lane} < {lane_symbol}",
        f"    {value} = tl.load({pointers[values]} + {program} * {lane_symbol}"
        f" + {lane}, mask={mask}, other=0)",
        f"    tl.store({pointers[out]} + {base} + {lane}, {value}, mask={mask})",
    ]
    write_source = (f"# {node.op}\n@triton.jit\n"
                    f"def {name}({', '.join(parameter for parameter, _ in write_pairs)}):\n"
                    + "\n".join(lines) + "\n")
    _require_affine(write_source, name, graph)
    write_kernel = TritonKernel(
        name=name,
        kind="store_at",
        ops=(node.op,),
        operands=operands,
        results=results,
        parameters=tuple(parameter for parameter, _ in write_pairs),
        launch=tuple(argument for _, argument in write_pairs),
        grid=f"({rows},)",
        source=write_source,
        launch_options=(("num_warps", config.num_warps),),
        store_at_contract=StoreAtContract(
            kernel=name,
            dst=dst,
            offsets=offsets,
            values=values,
            lane_count=lane_count,
            dst_elements=dst_elements,
            coverage=coverage_guard,
            trusted=trusted,
        ),
    )

    if static_coverage:
        return write_kernel

    # launch 1: copy the destination through, so untouched regions keep their
    # values. Its store address is `out_ptr + arange` -- affine as well, which
    # `_require_affine` checks for the same reason as the write kernel.
    copy_name = f"{name}_copy"
    dst_pointer = names.fresh(identifier(dst) + "_ptr")
    out_pointer = names.fresh(identifier(out) + "_ptr")
    copy_pairs = [(dst_pointer, Argument("tensor", dst)),
                  (out_pointer, Argument("tensor", out))]
    total_symbol = names.fresh("N")
    copy_block = names.fresh("BLOCK")
    copy_pairs.append((f"{total_symbol}: tl.constexpr",
                       Argument("value", dst_elements)))
    copy_pairs.append((f"{copy_block}: tl.constexpr",
                       Argument("value", str(config.default_block_size))))
    element = names.fresh("e")
    keep = names.fresh("keep")
    copy_lines = [
        f"    {element} = tl.program_id(0) * {copy_block} + tl.arange(0, {copy_block})",
        f"    {keep} = {element} < {total_symbol}",
        f"    tl.store({out_pointer} + {element}, "
        f"tl.load({dst_pointer} + {element}, mask={keep}, other=0), mask={keep})",
    ]
    copy_source = (f"# {node.op} (copy)\n@triton.jit\n"
                   f"def {copy_name}({', '.join(parameter for parameter, _ in copy_pairs)}):\n"
                   + "\n".join(copy_lines) + "\n")
    _require_affine(copy_source, copy_name, graph)
    copy_kernel = TritonKernel(
        name=copy_name,
        kind="store_at",
        ops=(node.op,),
        operands=operands,
        results=results,
        parameters=tuple(parameter for parameter, _ in copy_pairs),
        launch=tuple(argument for _, argument in copy_pairs),
        grid=f"(triton.cdiv({dst_elements}, {config.default_block_size}),)",
        source=copy_source,
        launch_options=(("num_warps", config.num_warps),),
        launch_guard=coverage_guard,
    )
    return (copy_kernel, write_kernel)


def host_guards(contracts, parameters):
    """Emit the fail-closed `store_at` checks for the generated host.

    Emitted before any allocation or launch, exactly as the contract says:

    * the offsets values are read once (`detach().cpu().tolist()` -- one D2H +
      sync) and checked for `0 <= base` / `base + lane_count <= dst.numel()`
      and pairwise disjointness. A small operator (< ~2 MB) can spend more time
      in this sync than in the kernel; that cost is a measured open item
      (`docs/STORE_AT_DESIGN_20261022.md` §3), not a silent trade.
    * `store_at_trusted` is the author-level declaration: the D2H checks are
      skipped and the declaration is recorded in the emitted source, so what
      was not checked stays visible.
    * a destination that is a declared input and has no copy kernel (the runs
      provably tile it) still has to be contiguous, so its layout check lives
      here; every other input's contiguity is already checked by the generic
      per-input guard because `store_at` carries no layout metadata.

    `parameters` maps declared input names to host variable names.
    """
    lines = []
    for contract in contracts:
        offsets_variable = parameters.get(contract.offsets)
        if offsets_variable is None:
            raise CompileError(
                DiagnosticKind.INTERNAL,
                f"store_at offsets '{contract.offsets}' is not a declared "
                f"input; the host has no table to validate",
            )
        if contract.trusted:
            lines += [
                f"    # store_at trusted_offsets author declaration for "
                f"'{contract.offsets}': bases are declared disjoint and in "
                f"bounds; the per-call D2H bounds/overlap check is skipped",
            ]
        else:
            values = f"_fg_sa_{contract.kernel}"
            lane = f"({contract.lane_count})"
            lines += [
                f"    # store_at caller contract: fail-closed bounds/overlap "
                f"check of '{contract.offsets}' (one D2H + sync)",
                f"    {values} = {offsets_variable}.detach().cpu().tolist()",
                f"    if min({values}) < 0 or max({values}) + {lane} > "
                f"({contract.dst_elements}):",
                f"        raise ValueError('store_at offsets out of bounds for "
                f"the destination: {contract.offsets}')",
                f"    {values}.sort()",
                f"    for _fg_i in range(len({values}) - 1):",
                f"        if {values}[_fg_i + 1] - {values}[_fg_i] < {lane}:",
                f"            raise ValueError('store_at offsets overlap; bases "
                f"must be pairwise disjoint')",
            ]
    return lines


def _require_affine(source, kernel_name, graph):
    """Contract 3: a store_at store address must classify as affine.

    Unconditional -- this is the operation's definition, not a platform policy.
    The platform policy (`affine_write_addresses`) runs the same classifier again
    over the whole kernel when the target asks for it.
    """
    from compiler.analysis.write_address import violations
    problems = violations(source, kernel_name)
    if problems:
        raise CompileError(
            DiagnosticKind.OPERATOR,
            "store_at lowering produced a non-affine write address: "
            + "; ".join(problems),
            pipeline=graph.pipeline,
        )


def reference_kernel(graph, group, memory, name):
    """Independent Python reference: copy, then write each segment.

    Mirrors the contract's failure modes (out of bounds, overlapping bases) so
    the reference oracle refuses the same programs the emitted host refuses.
    """
    from compiler.backend.reference import _dimension_parameters, _kernel

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    dst, offsets, values = operands
    out = results[0]
    names = Names(graph)
    variables = {value: names.reserve(identifier(value))
                 for value in operands + results}
    pairs = [(variables[value], Argument("tensor", value))
             for value in operands + results]
    _dimension_parameters(names, pairs,
                          [d for v in operands + results
                           for d in graph.values[v].shape])
    lane_count = product(graph.values[values].shape[1:])
    dst_elements = product(graph.values[dst].shape)
    rows = product([graph.values[values].shape[0]])
    lines = [
        f"    {variables[out]}[:] = {variables[dst]}[:]",
        "    _seen = []",
        f"    for _p in range({rows}):",
        f"        _base = {variables[offsets]}[_p]",
        f"        if _base < 0 or _base + ({lane_count}) > ({dst_elements}):",
        "            raise ValueError('store_at offsets out of bounds')",
        "        for _other in _seen:",
        f"            if _other < _base + ({lane_count}) and _base < _other + ({lane_count}):",
        "                raise ValueError('store_at offsets overlap; bases must be pairwise disjoint')",
        "        _seen.append(_base)",
        f"        for _j in range({lane_count}):",
        f"            {variables[out]}[_base + _j] = {variables[values]}[_p * ({lane_count}) + _j]",
    ]
    return _kernel(name, "store_at", group, operands, results, pairs, lines)
