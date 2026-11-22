"""FlowGPU compiler package.

Public surface:

    from flowgpu.compiler import compile_source, CompileError

Everything else in this package is an internal stage.

The public names are loaded lazily (PEP 562) so that importing a
frontend-neutral subpackage such as ``flowgpu.compiler.core`` does not drag in the DSL
lexer/parser or Lark.  This is part of the de-DSL boundary: core must be usable
without the text frontend being importable.
"""

__version__ = "0.26.0"

__all__ = [
    "__version__",
    "CompileError",
    "CompileResult",
    "compile_source",
    "compile_pipelines",
]


def __getattr__(name):
    if name in ("CompileError", "CompileResult", "compile_source", "compile_pipelines"):
        from flowgpu.compiler.compile import (
            CompileError,
            CompileResult,
            compile_pipelines,
            compile_source,
        )

        globals().update(
            CompileError=CompileError,
            CompileResult=CompileResult,
            compile_source=compile_source,
            compile_pipelines=compile_pipelines,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
