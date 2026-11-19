"""Access shape: the dimension a byte-traffic comparison cannot see.

§21.9's metric is `A = moved_bytes / floor_bytes`. Two plans that move the same
bytes but touch memory with a different *shape* are therefore indistinguishable
to it -- and before this module the layout comparison in
`analysis/layout_candidates` reported `preferred_by_traffic` as though that
settled the choice.

It does not settle it. Measured 2026-10-20 on the same operator and the same
byte volume:

  * the flat-tiled and per-plane-tiled lowerings differed 4.8%-13.5% on an H100
    (identical bytes, different lane utilisation and run lengths);
  * a per-lane gather read 1.81x on 国际通用 A and 0.02x on 昆仑芯 -- one chip
    collapses 50x on the same byte count.

So this module makes the missing dimension explicit. What it is: a *descriptor*
of how an access is shaped, and where that description came from. What it is
not: a device model. It does not predict cache behaviour, HBM transactions or
time -- §21.9 forbids those claims, and the descriptor's job is narrower:

    a plan comparison must never report two plans as equivalent when their
    access shapes differ.  When the byte model and the access descriptor point
    at different plans, the comparison is *under-determined* and says so.

The descriptors are derived from what the compiler actually has: axis-identity
contracts (`ir/index_maps.permutation_of`), the restricted view contract
(`ir/view_ir.ViewSpec`, when one is carried), and operator identity. The
preparatory index/scatter contracts are deliberately not consulted, because
nothing in the pipeline populates them yet -- deriving from them would be a
description of an unused IR.
"""
from dataclasses import dataclass

# Operators whose reads are addressed by a data-dependent index, so the access
# cannot be a contiguous run whatever the layout is.
# `gather_last` is a data-dependent read on the last axis, so it belongs
# here for the same reason the row gathers do: the address is not a
# function of the lane index.
GATHER_OPS = ("gather_rows", "gather_row_offset", "gather_last")
# Operators whose writes are addressed by a data-dependent index.
SCATTER_OPS = ("store_cache_4d", "scatter_by_index")
SEGMENT_OPS = ("segmented_matmul", "segmented_lora_matmul")
# Operators that only move axes, so the consumed layout is a permutation of the
# produced one.
PERMUTE_OPS = ("transpose", "move_first_last")
# A static patch geometry permutes elements, not axes: the source address is
# a permutation of the output index, so there is no contiguous run.
PATCH_OPS = ("patchify", "unpatchify")

BLOCK = "block"
STRIDED = "strided"
GATHER = "gather"
SCATTER = "scatter"
PERMUTED = "permuted"
#: A contiguous run written at a program-uniform device-table base. The base is
#: data-dependent but uniform across the lanes of one program, so the write is
#: `scalar_base + affine(arange)` -- the shape the measured targets sustain, as
#: opposed to SCATTER's arbitrary per-lane destination. `store_at` is this
#: shape. Kept a distinct kind so a future gate cannot confuse the two.
AFFINE_WRITE = "affine_write"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class AccessShape:
    """How one access touches memory, and where the description came from."""

    kind: str
    #: longest contiguous run, in elements, along the fastest moving axis.
    #: None means "not derivable statically", never "zero".
    run: int | None
    #: element stride along the axis the lanes advance on, when known.
    stride: int | None = None
    data_dependent: bool = False
    #: provenance: the IR fact this description was derived from.
    reason: str = ""

    def __post_init__(self):
        if self.kind not in (BLOCK, STRIDED, GATHER, SCATTER, PERMUTED,
                             AFFINE_WRITE, UNKNOWN):
            raise ValueError(f"unknown access shape kind: {self.kind}")
        if self.run is not None and self.run < 0:
            raise ValueError("a contiguous run cannot be negative")
        if self.data_dependent and self.kind not in (GATHER, SCATTER,
                                                    AFFINE_WRITE):
            raise ValueError(
                "a data-dependent access must be gather, scatter or an "
                "affine write at a program-uniform base"
            )


def _literal(value):
    """An integer extent, or None when the extent is symbolic."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def of_view(spec):
    """Descriptor for a restricted view (`ir/view_ir.ViewSpec`)."""
    last = _literal(spec.shape[-1])
    if spec.contiguous:
        return AccessShape(BLOCK, last, 1, False, "ViewSpec.contiguous")
    if len(spec.strides) > 1:
        inner = _literal(spec.strides[-1])
        if inner == 1:
            return AccessShape(BLOCK, last, 1, False, "ViewSpec strides[-1] == 1")
    stride = _literal(spec.strides[-1]) if spec.strides else None
    return AccessShape(STRIDED, 1, stride, False, "ViewSpec strided")


def of_op(op, attributes=None):
    """Descriptor for one graph operator's own access, from operator identity."""
    if op in SEGMENT_OPS:
        return AccessShape(GATHER, 1, None, True,
                           f"operator {op} selects its weight by segment id")
    if op in GATHER_OPS:
        return AccessShape(GATHER, 1, None, True, f"operator {op} reads by index")
    if op in SCATTER_OPS:
        return AccessShape(SCATTER, 1, None, True, f"operator {op} writes by index")
    if op in PATCH_OPS:
        return AccessShape(PERMUTED, 1, None, False,
                           f"operator {op} permutes elements by a static patch map")
    if op in PERMUTE_OPS:
        return AccessShape(STRIDED, 1, None, False, f"operator {op} permutes axes")
    if op in {"store_at", "store_at_trusted"}:
        # The base comes from a device table, so the access is data-dependent;
        # the lane run itself is one contiguous run whose address is
        # `scalar_base + affine(arange)`. That combination is what the measured
        # targets accept, which is why it is not SCATTER.
        return AccessShape(
            AFFINE_WRITE, None, 1, True,
            f"operator {op} writes a contiguous run at a program-uniform "
            f"device-table base",
        )
    return AccessShape(BLOCK, None, None, False, f"operator {op} is axis-local")


def worst(shapes):
    """The most restrictive descriptor in a plan.

    A plan is as good as its worst access: one gather in a group of otherwise
    contiguous accesses is what a backend with a gather penalty will charge.
    """
    rank = {BLOCK: 0, AFFINE_WRITE: 1, STRIDED: 1, PERMUTED: 2, GATHER: 2,
            SCATTER: 2, UNKNOWN: 3}
    return max(shapes, key=lambda shape: rank[shape.kind]) if shapes else AccessShape(UNKNOWN, None)


def of_reduction(reduction_axis, rank, last_extent, requires_contiguous):
    """Descriptor for the two arms of `analysis/layout_candidates`.

    Reducing along the last axis walks contiguous elements; reducing along any
    earlier axis walks with a stride, so each lane's useful run is one element
    and the transpose arm pays a materialisation to recover a contiguous walk.
    """
    last = _literal(last_extent)
    if reduction_axis == rank - 1:
        return AccessShape(BLOCK, last, 1, False, "reduction along the last axis")
    if requires_contiguous:
        return AccessShape(STRIDED, 1, None, False, "reduction along a non-last axis")
    return AccessShape(STRIDED, 1, None, False, "reduction along a non-last axis")


def differs(left, right):
    """Whether two plans' shapes disagree in a way a byte model cannot see."""
    if left.kind != right.kind:
        return True
    if left.run is None or right.run is None:
        # A plan whose run is not statically derivable is not thereby equal to
        # one that is: treat the comparison as undecided, which `underdetermined`
        # reports rather than silently resolving.
        return left.run != right.run
    return left.run != right.run or left.data_dependent != right.data_dependent


def underdetermined(byte_preference, access_preference):
    """The comparison verdict, when the two models may disagree.

    Returning "underdetermined" is the whole point: it is the state the old
    comparison could not represent, and it is a *correct* answer -- the choice
    needs either a device model we deliberately do not have (§21.9) or a
    measurement under the pre-registered protocol.
    """
    if byte_preference is None or access_preference is None:
        return "underdetermined"
    return "agree" if byte_preference == access_preference else "underdetermined"


def of_plan(kernels):
    """Access descriptors for a lowered plan, from the IR alone.

    Every kernel carries its operators, and an operator's access shape is fixed
    by its identity, so a plan's addressing can be judged *before* any text
    exists. That matters because the text-level rule saturates: a tile and a
    permutation are written with the same `idx // D * s0 + idx % D * s1` shape,
    and only the IR knows whether the recombined order is the output order.
    """
    shapes = []
    for kernel in kernels:
        for op in getattr(kernel, "ops", ()) or ():
            shapes.append(of_op(op))
    return shapes


def plan_shape(kernels):
    """The binding descriptor of a plan: its worst access."""
    return worst(of_plan(kernels))


def group_shape(fusion):
    """``plan_shape`` for a fusion plan: the binding shape, before emission.

    The post-lowering form has a trap that makes the pre-emission one
    authoritative: a kernel the target refuses is not in ``program.kernels``, so
    it is also not in that list's shape. Task154's plan on 昆仑芯 is the worked
    example -- `patchify` is refused by the kind whitelist, the emitted subset is
    the `mul` kernel alone, and the post-lowering shape therefore reads `block`
    while the plan's addressing is a permutation. Judging the plan is the only
    form that cannot be weakened by a refusal happening first.
    """
    return worst(tuple(of_op(node.op)
                       for group in fusion.groups for node in group.members))


def group_conflicts(fusion, platform):
    """``platform_conflicts`` for a *fusion plan*, i.e. before emission.

    This is the pre-emission form of the same judgement. `platform_conflicts`
    takes lowered kernels, which only exist after `lower()` has emitted a kernel
    per group; by then the text exists and the target has already been handed a
    plan it was measured not to sustain. The fusion report is the last point at
    which the plan is fully decided and no source has been written, so the gate
    belongs here.

    A group's access shape is decided by the operators it evaluates, so the
    groups are projected onto the one attribute the analysis reads (`ops`) rather
    than the whole node. Nothing else about a group can change the answer --
    which is what makes this a projection and not a second analysis.
    """
    return platform_conflicts(
        tuple(_PlannedGroup(f"group{number}", tuple(node.op for node in group.members))
              for number, group in enumerate(fusion.groups)),
        platform,
    )


@dataclass(frozen=True)
class _PlannedGroup:
    """One fusion group seen as a kernel, for the access-shape judgement."""

    name: str
    ops: tuple


def platform_conflicts(kernels, platform):
    """Feasibility messages derived from the IR, before emission.

    Additive and read-only by itself: it reports what the target is measured not
    to sustain and changes nothing. `contracts.violations` is the text-level
    check on the *emitted* source; this one needs no text, so it can run before
    the backend writes a line -- which is what `compile_source(platform_gate=)`
    uses it for. The two are meant to be read together: this one knows the shape,
    that one knows the expressions.

    Note the asymmetry with the pre-emission form: passed a lowered plan this
    sees only the kernels that survived the target's refusals, so a refused
    offending kernel makes the remaining subset look legal. `group_conflicts` is
    the form to gate on; this one is for a plan that was actually emitted.
    """
    found = []
    shape = plan_shape(kernels)
    honoured = getattr(platform, "gather_load_ok", True)
    if honoured:
        return found
    if shape.kind == PERMUTED:
        found.append(
            f"{platform.name}: the plan's access is a static element permutation "
            f"({shape.reason}); iterate the source densely and move the permutation "
            f"into the store address"
        )
    elif shape.kind == GATHER:
        found.append(
            f"{platform.name}: the plan's binding access is a data-dependent gather "
            f"({shape.reason}); iterate the source densely and move the permutation "
            f"into the store address"
        )
    elif shape.kind == SCATTER:
        found.append(
            f"{platform.name}: the plan's binding access is an indexed scatter "
            f"({shape.reason}); iterate the destination densely and move the "
            f"permutation into the load address"
        )
    return found
