"""FlowGPU public package namespace.

The compiler implementation currently lives in the top-level ``compiler``
package for historical reasons.  New code should import the public API from
``flowgpu``; the legacy ``compiler`` imports remain supported while the
implementation is moved behind this namespace.
"""
from compiler import (
    CompileError,
    CompileResult,
    __version__,
    compile_pipelines,
    compile_source,
)

__all__ = [
    "CompileError",
    "CompileResult",
    "__version__",
    "compile_pipelines",
    "compile_source",
]
