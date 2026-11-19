"""The access-shape dimension: what the byte-traffic model structurally cannot see.

The design flaw these tests pin: `analysis/layout_candidates` compared two plans
with `traffic_elements` alone and reported `preferred_by_traffic` as if it had
settled the choice. Two plans can move identical bytes and still differ 4.8-13.5%
on an H100, and 50x on 昆仑芯, because of how they touch memory. A comparison
that cannot represent "these are not equivalent" is wrong even when its byte
arithmetic is right.
"""
import pytest

from flowgpu.compiler.analysis.access_shape import (
    BLOCK,
    GATHER,
    SCATTER,
    STRIDED,
    UNKNOWN,
    AccessShape,
    differs,
    of_op,
    of_reduction,
    of_view,
    underdetermined,
    worst,
)
from flowgpu.compiler.analysis.layout_candidates import compare_layout_reduction
from flowgpu.compiler.ir.view_ir import ViewSpec


def test_a_contiguous_view_is_a_block_and_the_run_is_its_last_extent():
    spec = ViewSpec("kv", ("2", "64"), ("64", "1"), contiguous=True)
    shape = of_view(spec)
    assert shape.kind == BLOCK and shape.run == 64 and shape.stride == 1
    assert "ViewSpec.contiguous" in shape.reason


def test_a_strided_view_has_run_one_whatever_its_extent():
    """Packed-KV slices advance by a stride, so no run of consecutive lanes
    touches consecutive elements -- the extent is irrelevant."""
    spec = ViewSpec("packed", ("2", "64"), ("192", "3"))
    shape = of_view(spec)
    assert shape.kind == STRIDED and shape.run == 1
    assert shape.stride == 3


def test_gather_and_scatter_are_data_dependent_and_named_as_such():
    assert of_op("gather_rows").kind == GATHER
    assert of_op("gather_rows").data_dependent
    assert of_op("store_cache_4d").kind == SCATTER
    assert of_op("store_cache_4d").data_dependent
    assert of_op("silu").kind == BLOCK
    assert not of_op("silu").data_dependent


def test_a_data_dependent_access_cannot_claim_a_contiguous_kind():
    with pytest.raises(ValueError, match='a data-dependent access must be gather, scatter or an affine write at a program-uniform base'):
        AccessShape(BLOCK, 8, 1, True, "contradiction")
    with pytest.raises(ValueError, match='a contiguous run cannot be negative'):
        AccessShape(BLOCK, -1, 1, False, "negative run")


def test_the_worst_access_in_a_plan_is_what_a_backend_charges():
    plan = [of_op("silu"), of_op("gather_rows"), of_op("add")]
    assert worst(plan).kind == GATHER
    assert worst([]).kind == UNKNOWN


def test_byte_equal_plans_with_different_shapes_are_not_equivalent():
    """The whole point: a byte model would call these two identical."""
    block = AccessShape(BLOCK, 64, 1, False, "contiguous row")
    one = AccessShape(BLOCK, 1, None, False, "one element per run")
    assert differs(block, one)
    assert not differs(block, AccessShape(BLOCK, 64, 1, False, "same shape, other provenance"))


def test_an_underivable_run_is_not_treated_as_equal_to_a_derived_one():
    derived = AccessShape(BLOCK, 64, 1, False, "derived")
    unknown = AccessShape(BLOCK, None, None, False, "symbolic")
    assert differs(derived, unknown)


def test_reduction_along_the_last_axis_walks_contiguously():
    assert of_reduction(2, 3, 8, False).kind == BLOCK
    assert of_reduction(1, 3, 8, False).kind == STRIDED


def test_the_verdict_is_underdetermined_when_the_two_models_disagree():
    assert underdetermined("direct", "direct") == "agree"
    assert underdetermined("direct", "transpose_then_reduce") == "underdetermined"
    assert underdetermined(None, "direct") == "underdetermined"


def test_a_non_last_reduction_is_reported_as_underdetermined_not_as_direct():
    """Before this change the comparison said `direct`, full stop. The byte
    arithmetic is unchanged and still prefers direct; what changed is that the
    comparison no longer claims that settles it."""
    comparison = compare_layout_reduction((2, 4, 8), reduction_axis=1)
    assert comparison.preferred_by_traffic == "direct"
    assert comparison.preferred_by_access == "transpose_then_reduce"
    assert comparison.verdict == "underdetermined"
    assert "do not settle the same arm" in comparison.rationale
    assert comparison.as_dict()["verdict"] == "underdetermined"


def test_a_last_axis_reduction_keeps_its_settled_verdict():
    comparison = compare_layout_reduction((2, 4, 8), reduction_axis=-1)
    assert comparison.preferred_by_traffic == "direct"
    assert comparison.preferred_by_access == "direct"
    assert comparison.verdict == "agree"


def test_a_symbolic_shape_stays_underdetermined():
    comparison = compare_layout_reduction(("B", "H", "D"), reduction_axis=2)
    assert comparison.preferred_by_traffic is None
    assert comparison.verdict == "underdetermined"


def test_a_plan_is_judged_from_the_ir_before_any_text_exists():
    """The text-level rule saturates: a tile and a permutation are written with
    the same shape. The IR knows the difference, so the feasibility judgement
    must be available there."""
    from types import SimpleNamespace

    from flowgpu.compiler.analysis.access_shape import of_plan, plan_shape, platform_conflicts
    from flowgpu.compiler.backend.platform_configs import PLATFORMS

    copy_plan = [SimpleNamespace(ops=("pad_cat",))]
    assert plan_shape(copy_plan).kind == BLOCK
    assert of_plan(copy_plan)[0].reason == "operator pad_cat is axis-local"
    assert platform_conflicts(copy_plan, PLATFORMS["kunlunxin"]) == []

    scatter_plan = [SimpleNamespace(ops=("store_cache_4d",))]
    assert plan_shape(scatter_plan).kind == SCATTER
    assert len(platform_conflicts(scatter_plan, PLATFORMS["kunlunxin"])) == 1
    assert platform_conflicts(scatter_plan, PLATFORMS["cuda"]) == []

    gather_plan = [SimpleNamespace(ops=("gather_rows",))]
    assert plan_shape(gather_plan).kind == GATHER
    assert len(platform_conflicts(gather_plan, PLATFORMS["kunlunxin"])) == 1
