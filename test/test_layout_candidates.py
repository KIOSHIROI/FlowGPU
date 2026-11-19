import pytest

from flowgpu.compiler.analysis.layout_candidates import compare_layout_reduction


def test_direct_layout_reduction_is_analysis_only_and_lower_traffic():
    comparison = compare_layout_reduction((2, 4, 8), reduction_axis=1)
    assert comparison.direct.reduction_axis == 1
    assert comparison.direct.materializations == 0
    assert comparison.transpose_then_reduce.materializations == 1
    assert comparison.direct.traffic_elements == 80
    assert comparison.transpose_then_reduce.traffic_elements == 208
    assert comparison.preferred_by_traffic == "direct"
    assert "lower modeled element traffic" in comparison.rationale


def test_negative_axis_is_normalized_without_changing_backend_contract():
    comparison = compare_layout_reduction((2, 4, 8), reduction_axis=-1)
    assert comparison.direct.reduction_axis == 2
    assert comparison.direct.output_elements == 8


def test_symbolic_shapes_do_not_get_fake_costs():
    comparison = compare_layout_reduction(("B", "H", "D"), reduction_axis=2)
    assert comparison.direct.input_elements is None
    assert comparison.preferred_by_traffic is None
    assert "symbolic" in comparison.rationale


@pytest.mark.parametrize("axis", [-4, 3, 1.0, True])
def test_invalid_axis_is_rejected(axis):
    with pytest.raises(ValueError, match='reduction axis is out of range'):
        compare_layout_reduction((2, 4, 8), reduction_axis=axis)


def test_comparison_is_serializable_for_diagnostics():
    data = compare_layout_reduction((2, 4, 8), reduction_axis=0).as_dict()
    assert data["direct"]["name"] == "direct"
    assert data["transpose_then_reduce"]["kernels"] == 2
