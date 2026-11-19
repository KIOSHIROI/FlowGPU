"""FlowGPU public package namespace.

The compiler implementation lives in ``flowgpu.compiler`` and the optional
migration CLI in ``flowgpu.migration``.  The former top-level ``compiler`` and
``migration`` packages were removed in 0.18.0; use the ``flowgpu`` namespace.
"""
from .compiler import (
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
