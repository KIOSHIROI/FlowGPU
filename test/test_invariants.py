"""Optional state invariants reject the semantic faults the guards miss."""
import pytest

from flowgpu.compiler.runtime.invariants import (
    check_cache_position, check_mask, check_valid)


def test_cache_position_requires_contiguity():
    check_cache_position(8, expected=8, slots=32)
    with pytest.raises(ValueError, match="skipped"):
        check_cache_position(10, expected=8, slots=32)
    with pytest.raises(ValueError, match="outside"):
        check_cache_position(99, expected=8, slots=32)


def test_valid_range_and_mask_prefix():
    assert check_valid(9, slots=32) == 9
    with pytest.raises(ValueError, match="valid"):
        check_valid(0, slots=32)
    check_mask([0] * 9 + [-1e9] * 23, valid=9)
    with pytest.raises(ValueError, match="written prefix"):
        check_mask([1] + [0] * 8 + [-1e9] * 23, valid=9)
    with pytest.raises(ValueError, match="mask must mask"):
        check_mask([0] * 32, valid=9)
