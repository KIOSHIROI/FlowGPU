"""Compatibility facade for the 21.6 dtype/precision rules.

The implementation moved to ``compiler.frontend.dtype_rules`` so the
frontend can use it without importing a layer above itself (review P2,
layering, 2026-10-26).  Existing imports of this module keep working.
"""

from compiler.core.diagnostic import DiagnosticKind, RuleError  # noqa: F401
from compiler.core.dtype_rules import (  # noqa: F401
    ACC_FLOOR,
    DTYPE_BYTES,
    DTYPE_CLASSES,
    NARROW_INT_DTYPES,
    STORAGE_DTYPES,
    check_accumulation,
    declared,
    declined,
    dtype_class,
    is_float,
    require_arithmetic,
    require_numeric,
    width,
)
