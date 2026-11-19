"""Compatibility re-export of the frontend-neutral diagnostics.

The classes moved to ``flowgpu.compiler.core.diagnostic`` in Phase 0 of the de-DSL
work.  Existing DSL code and tests keep importing from here; the identity is
the same object, so ``except CompileError`` and isinstance checks are unchanged.
"""
from flowgpu.compiler.core.diagnostic import (  # noqa: F401
    CompileError,
    DiagnosticKind,
    RuleError,
)

__all__ = ["CompileError", "DiagnosticKind", "RuleError"]
