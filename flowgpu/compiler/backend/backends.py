"""Shared backend classification and naming utilities.

RoPE is a pair-indexing anchor with explicit cosine/sine tensors. Reserved
operators are refused in the frontend. Both targets consume the same kernel
partition and conservative buffer plan; numerical agreement is tested separately.
"""

from dataclasses import dataclass

from flowgpu.compiler.optimization.fusion import is_operator
from flowgpu.compiler.core.diagnostic import CompileError, DiagnosticKind

# Graph/name helpers live in flowgpu/compiler/shared.py (review P2, layering): `ir`,
# `analysis` and `optimization` need them too, and they may not import backend.
from flowgpu.compiler.shared import (  # noqa: F401  (re-exported for compatibility)
    Names,
    ROW_WISE,
    group_operands,
    group_results,
    identifier,
    is_symbol,  # noqa: F401  (re-exported; was imported from here)
    product,
    scalar_operands,
    store_rotation_chain,
    tensor_operands,
    unique_formal,
    used_names,
)


@dataclass(frozen=True)
class LibraryDispatch:
    """One policy for fusion visibility and library emission.

    Always-library anchors must stay opaque. Conditional anchors can fuse,
    and use a library only when the resulting group has no fused region.
    This is a dispatch contract, not a claim that either route always wins.
    """

    always: tuple = ()
    unfused: tuple = ()

    @property
    def opaque_anchors(self):
        # Native attention is refused, and must not absorb pointwise work.
        return tuple(dict.fromkeys(("attention", "causal_attention") + self.always))

    def uses_library(self, group, *, pipeline=None):
        op = group.anchor.op if group.anchor is not None else None
        if op in self.always:
            if group.fused:
                raise CompileError(DiagnosticKind.INTERNAL,
                                   f"library anchor '{op}' contains a fused region",
                                   pipeline=pipeline)
            return True
        return op in self.unfused and not group.fused


def library_dispatch(gpu_backend="triton", library_matmul="unfused"):
    """Validate options and derive the policy used by both compiler stages."""
    if gpu_backend not in {"triton", "hybrid"}:
        raise ValueError("gpu_backend must be 'triton' or 'hybrid'")
    if library_matmul not in {"unfused", "all"}:
        raise ValueError("library_matmul must be 'unfused' or 'all'")
    if gpu_backend != "hybrid":
        if library_matmul != "unfused":
            raise ValueError("library_matmul only applies to gpu_backend='hybrid'")
        return LibraryDispatch()
    always = ("attention", "causal_attention", "rms_norm")
    if library_matmul == "all":
        return LibraryDispatch(always + ("matmul",))
    return LibraryDispatch(always, ("matmul",))


# rms_norm's eps. This is a semantic choice, not a target's: both backends
# compute the same normalization, so they share the constant. Which values are
# *not* shared are the ones a target invents for itself -- Triton's block
# sizes, which have no meaning in Python.
RMS_EPS = 1e-6

TILED = ("matmul",)
PERMUTING = ("transpose",)

# The kernel kind each group of operators needs from a backend.
# The three host-lowered layout operators, named once for every consumer:
# fusion excludes them from hosting pointwise regions (there are no
# registers to fuse into on a copy), the Triton host emits their calls
# without a grid, and the reference lowers them as flat-list index remaps.
LAYOUT_OPS = ("reshape", "split", "concat", "interleave")
# `interleave` doubles the last axis by alternating two same-shaped
# operands; it exists so a per-word nibble shuffle can become a new axis
# (AWQ promotion, 2026-10-27).  Host-lowered like its siblings.

from flowgpu.compiler.backend.primitives import PRIMITIVES

KERNELS = {
    "primitive": PRIMITIVES,
    "rowwise": ROW_WISE,
    "matmul": TILED,
    "transpose": PERMUTING,
    "rope": ("rope",),
    "attention": ("attention", "causal_attention"),
    # Layout operators move elements rather than compute them, so neither
    # backend emits a device kernel for them: the lowering is host-side
    # (torch slicing/cat/reshape into the slot the plan already allocated).
    # They are still a kernel kind, because the partition must cover every
    # node and the platform gate must be able to name what it allows.
    "layout": LAYOUT_OPS,
    "cache_store": ("store_cache_4d",),
    "cache_write": ("cache_write",),
    "cache_write_hsd": ("cache_write_hsd",),
    "pad_cat": ("pad_cat",),
    # The scan kind holds the whole carried-state family: a prefix sum
    # carries one fp32 total, a prefix min/max carries (value, index).
    # The lowering reports which of them it emits (`scan_backend.SERVES`).
    "scan": ("cumsum", "cumsum_totals", "cummin_last", "cummax_last"),
    "patchify": ("patchify",),
    "segmented_matmul": ("segmented_matmul", "segmented_lora_matmul"),
    "scatter_by_index": ("scatter_by_index",),
    # A contiguous run written at a device-table base: the write is affine
    # (scalar base + arange), unlike scatter_by_index's arbitrary per-lane
    # destination. See docs/STORE_AT_DESIGN_20261022.md.
    "store_at": ("store_at", "store_at_trusted"),
    "topk": ("topk_last",),
    "sort": ("sort_last",),
    "mean": ("mean_last",),
    "layer_norm": ("layer_norm",),
    "conv2d": ("conv2d",),
    "conv2d_same": ("conv2d_same",),
    "conv2d_stride2": ("conv2d_stride2",),
    "conv2d_stride2_same": ("conv2d_stride2_same",),
    "conv2d_ex": ("conv2d_ex",),
    "axis_gather": ("gather_last",),
    "reduce_with_index": ("argmax_last", "argmin_last"),
    "index_table": ("segment_rank",),
    "recurrence": ("gated_delta_recurrence",),
    "linear_decode": ("linear_decode_forward",),
}


# Operators no backend can lower, and why. v0.1 kept the reserved trio here;
# v0.1.2 lowered them as the `layout` host kernels, so the table is empty
# and stays: a future operator that neither backend can emit lands here with
# its reason, and refusal_for reads it so the classification stays complete
# (classification_gaps) while no operator is silently dropped.
REFUSED = {}


def kernel_kind(anchor) -> str:
    """Which lowering a group needs, from its anchor.

    A group with no anchor is all pointwise operators, which is what a
    pipeline of elementwise maths lowers to when there is nothing to fuse it
    into (13.6).
    """
    if anchor is None:
        return "pointwise"

    for kind, operators in KERNELS.items():
        if anchor.op in operators:
            return kind

    return "unknown"


def classification_gaps(operators, pointwise_expressions) -> set:
    """The operators a backend has no decision for.

    `operators` is the registry, `pointwise_expressions` the backend's table
    for pointwise operators. An operator missing from all of them would be a
    silent hole: 13.8 asks what it means, and nothing would answer.
    """
    from flowgpu.compiler.core.operators import AXIS_OPERATORS, COMPOSED_OPERATORS
    # Composed operators are classified as "no backend kernel, expanded before
    # lowering" (flowgpu/compiler/ir/composite_ops.py) rather than silently missing.
    classified = (set(pointwise_expressions) | set(REFUSED) | set(AXIS_OPERATORS)
                  | set(COMPOSED_OPERATORS))

    for operators_of_kind in KERNELS.values():
        classified |= set(operators_of_kind)

    return set(operators) - classified

def storage_of(memory, value) -> str | None:
    """A value's storage, or None when it needs none.

    None is a normal answer here -- 13.7 gives a fused or unread value no
    buffer -- so callers decide what it means rather than finding out by
    exception.
    """
    for buffer in memory.buffers:
        if buffer.value == value:
            return buffer.storage

    raise AssertionError(f"no buffer for {value!r}")


def allocation(memory, value) -> str:
    """Where a value lives: the caller's tensor, or the slot it shares."""
    storage = storage_of(memory, value)

    if storage == "input":
        return identifier(value)
    if storage is not None:
        return f"slot{storage[4:]}"

    # A value with no storage cannot be read or written, so reaching here is a
    # compiler defect rather than something a program can cause.
    raise AssertionError(f"{value!r} has no storage and cannot be an argument")

@dataclass(frozen=True)
class Refusal:
    """A kernel a backend cannot emit, and why.

    The reason is a fact about the language, so both backends produce the same
    refusal; only the target named in the message differs.
    """

    operator: str
    values: tuple
    reason: str
    target: str = ""

    def message(self) -> str:
        values = ", ".join(self.values) if self.values else "no value"
        target = f" to {self.target}" if self.target else ""

        return f"cannot lower '{self.operator}' ({values}){target}: {self.reason}"


@dataclass(frozen=True)
class SkippedKernel:
    """A kernel no backend emitted, because nothing reads its result."""

    name: str
    values: tuple
    reason: str

    def message(self) -> str:
        return f"skipped {self.name} ({', '.join(self.values)}): {self.reason}"


def load_rotation_roots(graph, group) -> set:
    """Rotation nodes the sweep reads through, not values it must materialize.

    A ``move_first_last`` chain that bottoms out at a pipeline input feeds
    the group's reductions through readdressed loads (13.6), so none of its
    links needs a pointwise lowering or a refusal. A pure elementwise link
    (a cast, pointwise math) inside the chain does not break the walk: the
    sweep recomputes it at the mapped address, so rotations above it stay
    load-side foldable too.
    """
    from flowgpu.compiler.optimization.reduction_fusion import ELEMENTWISE_LINKS
    links = {node.output: node for node in group.members
             if node.op == 'move_first_last'
             or node.op in ELEMENTWISE_LINKS}
    roots = set()
    for node in links.values():
        if node.op != 'move_first_last':
            continue
        source = node.inputs[0]
        walked = {node.output}
        rotated = {node.output}
        while True:
            producer = links.get(source) or graph.producer_of(source)
            if producer is None or producer.op == 'input':
                roots |= rotated
                break
            if producer.output not in links or producer.output in walked:
                break
            walked.add(producer.output)
            if producer.op == 'move_first_last':
                rotated.add(producer.output)
            source = producer.inputs[0]
    return roots


def refusal_for(graph, group, pointwise_expressions, target) -> Refusal | None:
    """Why this group cannot be lowered, if it cannot.

    `pointwise_expressions` is the calling backend's own table, so a backend
    that forgets an operator is caught here rather than at emission time.
    """
    anchor = group.anchor

    # A multi-result opaque operator owns its whole group: its sibling node is
    # the same computation's second result, not a second operator to lower.
    if anchor is not None and anchor.op in {'store_cache_4d', 'linear_decode_forward',
                                            'topk_last', 'segment_rank',
                                            'argmax_last', 'argmin_last',
                                            'gated_delta_recurrence',
                                            'cummin_last', 'cummax_last',
                                            'cumsum_totals'}:
        return None
    if group.row_gather and all(node.op == 'gather_rows' for node in group.members):
        return None

    # `add_broadcast` maps its row operand by dropping the trailing axis, which
    # only the scan epilogue emitter does.  Everywhere else it would be lowered
    # as a same-index add, i.e. with a wrong (broadcast) address, so refuse by
    # name instead.
    if any(node.op == 'add_broadcast' for node in group.members):
        if anchor is None or anchor.op not in ('cumsum', 'cumsum_totals'):
            return Refusal(
                'add_broadcast',
                tuple(node.output for node in group.members
                      if node.op == 'add_broadcast' and node.output is not None),
                'a row-broadcast add is emitted inside a fused scan epilogue '
                'only; put it where the scan can fuse it',
                target)

    rotations = {node.output for node in store_rotation_chain(graph, group)}
    rotations |= load_rotation_roots(graph, group)
    if group.indexed:
        from flowgpu.compiler.optimization.indexed_fusion import supported
        # A `split` node only reaches an indexed group when the caller opted
        # into the zero-cost split lowering (indexed_fusion merges it there);
        # in the default path it stays a layout anchor in its own group.
        if all(supported(node.op) or node.op in {'split', 'concat', 'interleave', 'transpose'} or node.output in rotations
               for node in group.members):
            return None

    for node in group.members:
        if node.op in REFUSED:
            values = tuple(
                member.output
                for member in group.members
                if member.op == node.op and member.output is not None
            )

            return Refusal(node.op, values, REFUSED[node.op], target)

        if node.output in rotations:
            continue

        if node.op in {"cast_like", "cast_to", "to_fp32"} and node is not anchor:
            # A cast is the declared-dtype boundary itself: every emitter
            # applies it as the operand's rounding, so it needs no entry
            # in the pointwise table.
            continue

        if getattr(node.attributes, "program_scalar", None) is not None:
            # A program-uniform scalar is a producer folded into the kernel
            # (one scalar load per program), not an expression the pointwise
            # table spells. The emitters that can host it handle the node by
            # its IR attribute; a backend that cannot will refuse where the
            # address is formed.
            continue

        if not is_operator(node) or node is anchor:
            continue

        if node.op not in pointwise_expressions:
            return Refusal(
                node.op,
                (node.output,) if node.output else (),
                "no pointwise lowering is registered for this operator",
                target,
            )

    if anchor is not None and kernel_kind(anchor) == "unknown":
        return Refusal(
            anchor.op,
            (anchor.output,) if anchor.output else (),
            "no lowering is registered for this operator",
            target,
        )

    return None

def comment(value, group) -> str:
    """What a value is, in the generated code's own vocabulary.

    Both backends write the same sentence, so their output can be read side by
    side, which is how the second one checks the first.
    """
    if value in group.fused_values:
        return f"{value}: fused, 13.7 keeps it in registers"

    return str(value)


@dataclass(frozen=True)
class Argument:
    """One argument of a generated kernel call.

    A `tensor` argument names a FlowGPU value and is resolved through the 13.7
    plan, which is what decides whether it is the caller's tensor or a slot.
    A `scalar` argument also names a FlowGPU value, but the value is a rank-0
    input: it reaches the host as a plain number, so the host passes that
    number itself and never asks the plan for storage.
    A `value` argument is an expression the host evaluates itself: a
    dimension, a product, a block size. `shape`/`stride` metadata uses
    "value:axis", and `contiguous` names the tensor whose layout is queried.
    `adjacent_views` names two statically same-typed/shaped external values as
    "left:right"; the host rechecks their storage/stride relationship per call.
    """

    kind: str
    text: str
