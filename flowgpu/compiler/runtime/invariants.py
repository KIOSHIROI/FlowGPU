"""Optional host-side invariants for cache/mask state.

The compiler's generated guards cover declarations. These helpers cover the
state semantics that the audit sheet can only flag: cache positions must be
contiguous, a mask must match the written prefix, and a valid count must be in
range. They are opt-in; a caller that has already proven these facts pays
nothing.
"""
from __future__ import annotations


def check_cache_position(loc, *, expected, slots):
    """Reject a loc that is out of range or skips the next expected slot."""
    if slots <= 0:
        raise ValueError("cache slots must be positive")
    if not 0 <= int(loc) < slots:
        raise ValueError("cache loc is outside [0, slots)")
    if int(loc) != int(expected):
        raise ValueError(f"cache position skipped: expected {expected}, got {loc}")


def check_valid(valid, *, slots):
    """Reject a valid count outside 1..slots."""
    value = int(valid)
    if not 1 <= value <= slots:
        raise ValueError("valid must be in [1, slots]")
    return value


def check_mask(mask, *, valid):
    """Reject a mask that does not match the written prefix.

    ``mask`` accepts a Python list or a one-dimensional tensor-like object
    supporting ``len`` and ``float()`` per element. The first ``valid``
    entries must be 0 and the rest must be at most ``-1e8``.
    """
    mask = list(mask)
    if len(mask) != 0 and valid > len(mask):
        raise ValueError("valid exceeds the mask length")
    for index, value in enumerate(mask):
        number = float(value)
        if index < valid and number != 0.0:
            raise ValueError("mask must be 0 for the written prefix")
        if index >= valid and number > -1e8:
            raise ValueError("mask must mask positions beyond valid")


__all__ = ["check_cache_position", "check_valid", "check_mask"]
