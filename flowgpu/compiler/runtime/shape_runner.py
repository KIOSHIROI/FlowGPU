"""Pick the right dynamic-shape strategy from the program's declarations.

Symbolic declarations use one compiled module for runtime extents. Literal
declarations cannot be recompiled with shape_bindings, so they use the
declaration-rewrite fallback. This is a dispatcher, not a third compilation
scheme; each backend is tested independently.
"""
from __future__ import annotations

from dataclasses import dataclass


def uses_symbolic_shapes(source: str, pipeline: str) -> bool:
    """Inspect declarations only: an operator may reject symbolic lowering."""
    import re
    if not re.search(rf"pipeline\s+{re.escape(pipeline)}\b", source):
        raise ValueError(f"no pipeline {pipeline!r}")
    for match in re.finditer(r"input\s+[A-Za-z_][A-Za-z0-9_]*\s*:\s*[A-Za-z_][A-Za-z0-9_]*\s*\[([^\]]*)\]",
                             source):
        dimensions = [item.strip() for item in match.group(1).split(",") if item.strip()]
        if any(not dimension.isdigit() for dimension in dimensions):
            return True
    return False


@dataclass
class ShapeRunner:
    source: str
    pipeline: str
    cache_dir: str
    options: dict | None = None

    def __post_init__(self) -> None:
        self._fallback_error = None
        from flowgpu.compiler.runtime.literal_shape import LiteralShapeRunner
        if uses_symbolic_shapes(self.source, self.pipeline):
            from flowgpu.compiler.runtime.symbolic_shape import SymbolicShapeRunner
            try:
                self._backend = SymbolicShapeRunner(
                    self.source, self.pipeline, self.cache_dir, self.options)
            except Exception as error:
                # An operator that cannot be emitted symbolically still has a
                # per-shape path: rewrite the declared extents and compile.
                self._fallback_error = error
                self._backend = LiteralShapeRunner(
                    self.source, self.pipeline, self.cache_dir, self.options)
        else:
            self._backend = LiteralShapeRunner(
                self.source, self.pipeline, self.cache_dir, self.options)

    @property
    def mode(self) -> str:
        return type(self._backend).__name__

    def __call__(self, *tensors):
        try:
            return self._backend(*tensors)
        except Exception as error:
            if type(self._backend).__name__ != "SymbolicShapeRunner":
                raise
            from flowgpu.compiler.runtime.literal_shape import LiteralShapeRunner
            self._fallback_error = error
            self._backend = LiteralShapeRunner(
                self.source, self.pipeline, self.cache_dir, self.options)
            return self._backend(*tensors)


__all__ = ["ShapeRunner", "uses_symbolic_shapes"]
