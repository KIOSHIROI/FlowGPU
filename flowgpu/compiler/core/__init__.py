"""Frontend-neutral primitives shared by every FlowGPU frontend.

Nothing in this package may import ``flowgpu.compiler.frontend``: the direction is
frontends -> core, never core -> a frontend.  Phase 0 of the de-DSL work moved
positions, diagnostics and the tensor type here; the DSL re-exports them so
existing imports keep working while the boundary becomes testable.
"""
from flowgpu.compiler.core.diagnostic import CompileError, DiagnosticKind, RuleError
from flowgpu.compiler.core.position import Position
from flowgpu.compiler.core.types import TensorType, format_shape, is_symbol

__all__ = ["CompileError", "DiagnosticKind", "RuleError", "Position",
           "TensorType", "format_shape", "is_symbol"]
