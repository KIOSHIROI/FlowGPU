import pytest

from compiler.ir.view_ir import PackedKVSlice, ViewSpec


def test_packed_kv_slice_is_explicit_read_only_view_contract():
    view = ViewSpec("packed", ("B", "S", "2", "H"), ("s0", "s1", "s2", "1"), contiguous=False)
    spec = PackedKVSlice("packed", axis=2, start=0, stop=1, value=view)
    assert spec.width == 1
    assert spec.as_dict()["strides"] == ["s0", "s1", "s2", "1"]
    assert spec.as_dict()["read_only"] is True


def test_negative_axis_is_normalized_and_offset_is_preserved():
    view = ViewSpec("kv", ("B", "S", "H"), ("s0", "s1", "1"), offset=7)
    spec = PackedKVSlice("kv", axis=-1, start=2, stop=5, value=view)
    assert spec.axis == 2
    assert spec.value.offset == 7


@pytest.mark.parametrize("bounds", [(-1, 2), (2, 2), (4, 7)])
def test_invalid_packed_slice_bounds_are_rejected(bounds):
    view = ViewSpec("kv", ("B", "S", "4"), ("s0", "s1", "1"))
    with pytest.raises(ValueError, match='packed-KV slice'):
        PackedKVSlice("kv", 2, *bounds, value=view)


def test_general_writable_view_is_not_part_of_this_contract():
    with pytest.raises(ValueError, match='restricted views are read-only'):
        ViewSpec("x", ("N",), ("1",), read_only=False)
