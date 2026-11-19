"""Compatibility facade for the future ``flowgpu.compiler`` namespace.

It re-exports the documented public API only; internal compiler modules are not
part of the compatibility contract.
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
