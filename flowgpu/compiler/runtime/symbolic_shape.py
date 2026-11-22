"""A compiled symbolic-shape module, used directly across extents.

FlowGPU's symbolic lowerings emit launch-time constexprs for dimension symbols,
so a program whose operators support runtime extents can be compiled once and
called with different shapes.  This wrapper makes that path explicit; it does
not add a per-shape compile cache.  Programs with literal-only operators still
need DynamicShapeRunner or an explicit specialization.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SymbolicShapeRunner:
    source: str
    pipeline: str
    cache_dir: str
    options: dict | None = None

    def __post_init__(self) -> None:
        from flowgpu.compiler.runtime.runtime import compile_pipeline
        self.options = dict(self.options or {})
        if "shape_bindings" in self.options:
            raise ValueError("SymbolicShapeRunner compiles without shape_bindings")
        self._run = compile_pipeline(
            self.source, pipeline=self.pipeline, cache_dir=self.cache_dir,
            **self.options)

    def __call__(self, *tensors):
        return self._run(*tensors)


__all__ = ["SymbolicShapeRunner"]
