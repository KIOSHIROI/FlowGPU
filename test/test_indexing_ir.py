import pytest

from flowgpu.compiler.ir.indexing_ir import IndexSpec, ScatterWriteSpec


def test_index_spec_makes_mask_and_bounds_policy_explicit():
    spec = IndexSpec("kv", "positions", axis=1, result_shape=("B", "T", "H"), masked=True, bounds_check="mask")
    assert spec.masked is True
    assert spec.bounds_check == "mask"


def test_scatter_write_requires_explicit_conflict_and_atomic_contract():
    index = IndexSpec("out", "positions", axis=1, result_shape=("B", "T", "H"))
    spec = ScatterWriteSpec("out", index, "values", conflict="add", atomic=True)
    assert spec.as_dict()["atomic"] is True


def test_may_alias_rejects_implicit_last_writer_wins():
    index = IndexSpec("out", "positions", axis=0, result_shape=("N",))
    with pytest.raises(ValueError, match='may-alias scatter needs an explicit conflict policy'):
        ScatterWriteSpec("out", index, "values", aliasing="may_alias")


@pytest.mark.parametrize("conflict", ["add", "max"])
def test_reduction_scatter_requires_atomicity(conflict):
    index = IndexSpec("out", "positions", axis=0, result_shape=("N",))
    with pytest.raises(ValueError, match='reduction scatter conflicts require atomic=True'):
        ScatterWriteSpec("out", index, "values", conflict=conflict)


def test_mask_policy_cannot_be_hidden_in_unmasked_index():
    with pytest.raises(ValueError, match='mask bounds policy requires masked indexing'):
        IndexSpec("x", "i", axis=0, result_shape=("N",), bounds_check="mask")
