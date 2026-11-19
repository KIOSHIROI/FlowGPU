"""Analysis-only layout candidates for cross-layout reductions.

The comparator deliberately does not select a backend or rewrite the graph. It
makes the tradeoff between reducing in the current layout and materializing a
transposed layout explicit so a later measured policy can consume it.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class LayoutReductionCandidate:
    name: str
    reduction_axis: int
    input_elements: int | None
    output_elements: int | None
    temporary_elements: int | None
    reads: int | None
    writes: int | None
    kernels: int
    materializations: int
    requires_contiguous: bool
    notes: tuple = ()

    @property
    def traffic_elements(self):
        if self.reads is None or self.writes is None:
            return None
        return self.reads + self.writes


@dataclass(frozen=True)
class LayoutReductionComparison:
    direct: LayoutReductionCandidate
    transpose_then_reduce: LayoutReductionCandidate
    preferred_by_traffic: str | None
    rationale: str
    #: Which arm the *access shape* prefers, from `analysis.access_shape`.
    #: The byte model cannot see this dimension at all, so when the two
    #: preferences disagree the comparison is under-determined rather than
    #: settled -- see `verdict`.
    preferred_by_access: str | None = None
    #: "agree" when both models pick the same arm; "underdetermined" when they
    #: disagree or when one of them cannot decide. Under-determined is a correct
    #: answer: it is what a byte-only model used to hide.
    verdict: str = "underdetermined"

    def as_dict(self):
        def encode(candidate):
            return {
                "name": candidate.name,
                "reduction_axis": candidate.reduction_axis,
                "input_elements": candidate.input_elements,
                "output_elements": candidate.output_elements,
                "temporary_elements": candidate.temporary_elements,
                "reads": candidate.reads,
                "writes": candidate.writes,
                "traffic_elements": candidate.traffic_elements,
                "kernels": candidate.kernels,
                "materializations": candidate.materializations,
                "requires_contiguous": candidate.requires_contiguous,
                "notes": list(candidate.notes),
            }
        return {
            "direct": encode(self.direct),
            "transpose_then_reduce": encode(self.transpose_then_reduce),
            "preferred_by_traffic": self.preferred_by_traffic,
            "preferred_by_access": self.preferred_by_access,
            "verdict": self.verdict,
            "rationale": self.rationale,
        }


def _elements(shape):
    result = 1
    for dimension in shape:
        text = str(dimension)
        if not text.isdigit():
            return None
        result *= int(text)
    return result


def compare_layout_reduction(
    source_shape,
    *,
    reduction_axis,
    output_shape=None,
    layout="row_major",
    direct_contiguous=False,
):
    """Compare direct reduction with transpose/materialize/reduce.

    This is an analysis contract, not a lowering decision. ``None`` is used for
    symbolic sizes instead of inventing a cost. The traffic model counts tensor
    element reads/writes and assumes one input read plus one output write for a
    direct reduction. The alternate path adds one full materialization read and
    write, then reduces the materialized tensor.
    """
    shape = tuple(str(d) for d in source_shape)
    rank = len(shape)
    if rank == 0:
        raise ValueError("layout reduction needs a non-empty source shape")
    if type(reduction_axis) is not int or not -rank <= reduction_axis < rank:
        raise ValueError("reduction axis is out of range")
    axis = reduction_axis % rank
    if output_shape is None:
        output = shape[:axis] + shape[axis + 1:]
    else:
        output = tuple(str(d) for d in output_shape)
        if len(output) != rank - 1:
            raise ValueError("reduction output rank must be source rank minus one")
    input_elements = _elements(shape)
    output_elements = _elements(output)
    direct = LayoutReductionCandidate(
        "direct",
        axis,
        input_elements,
        output_elements,
        None,
        input_elements,
        output_elements,
        1,
        0,
        bool(direct_contiguous),
        ("current layout is reduced directly",),
    )
    transpose_reads = input_elements
    transpose_writes = input_elements
    reduce_reads = input_elements
    reduce_writes = output_elements
    transpose = LayoutReductionCandidate(
        "transpose_then_reduce",
        axis,
        input_elements,
        output_elements,
        input_elements,
        None if None in (transpose_reads, reduce_reads) else transpose_reads + reduce_reads,
        None if None in (transpose_writes, reduce_writes) else transpose_writes + reduce_writes,
        2,
        1,
        True,
        ("materialize a reduction-friendly layout", "backend may reuse the transposed buffer"),
    )
    if direct.traffic_elements is None or transpose.traffic_elements is None:
        preferred = None
        rationale = "symbolic shape prevents a traffic-only preference"
    elif direct.traffic_elements < transpose.traffic_elements:
        preferred = direct.name
        rationale = "direct reduction has lower modeled element traffic"
    elif transpose.traffic_elements < direct.traffic_elements:
        preferred = transpose.name
        rationale = "transpose path has lower modeled element traffic"
    else:
        preferred = None
        rationale = "modeled element traffic ties; compare layout constraints and device timing"
    # The second dimension: reducing along the last axis walks contiguous
    # elements; reducing along any earlier axis walks with a stride, so the
    # direct arm's run is one element while the transpose arm buys a contiguous
    # walk for one extra materialisation. Byte traffic alone cannot compare
    # those two, so the descriptor decides only what it can.
    from compiler.analysis.access_shape import of_reduction, underdetermined

    direct_shape = of_reduction(axis, rank, shape[-1], direct_contiguous)
    transpose_shape = of_reduction(rank - 1, rank, shape[-1], True)
    access_preferred = direct.name if direct_shape == transpose_shape else transpose.name
    verdict = underdetermined(preferred, access_preferred)
    if verdict == "underdetermined":
        rationale = (
            f"{rationale}; byte traffic and access shape do not settle the same arm "
            f"(traffic prefers {preferred!r}, access prefers {access_preferred!r}), so "
            f"the choice needs a device model this compiler deliberately does not "
            f"claim (§21.9) or a pre-registered measurement"
        )
    return LayoutReductionComparison(direct, transpose, preferred, rationale,
                                     access_preferred, verdict)
