"""Pick the right dynamic-shape strategy from the program's declarations.

Symbolic declarations use one compiled module for runtime extents. Literal
declarations cannot be recompiled with shape_bindings, so they use the
declaration-rewrite fallback. This is a dispatcher, not a third compilation
scheme; each backend is tested independently.
"""
from __future__ import annotations

from dataclasses import dataclass


def uses_symbolic_shapes(source: str, pipeline: str) -> bool:
    from flowgpu.compiler.compile import compile_source
    compiled = compile_source(source, emit=False)
    for item in compiled.pipelines:
        if item.pipeline.name != pipeline:
            continue
        for node in item.dataflow.nodes:
            if node.op != "input":
                continue
            if any(not str(extent).isdigit() for extent in item.dataflow.values[node.output].shape):
                return True
        return False
    raise ValueError(f"no pipeline {pipeline!r}")


@dataclass
class ShapeRunner:
    source: str
    pipeline: str
    cache_dir: str
    options: dict | None = None

    def __post_init__(self) -> None:
        if uses_symbolic_shapes(self.source, self.pipeline):
            from flowgpu.compiler.runtime.symbolic_shape import SymbolicShapeRunner
            self._backend = SymbolicShapeRunner(
                self.source, self.pipeline, self.cache_dir, self.options)
        else:
            from flowgpu.compiler.runtime.literal_shape import LiteralShapeRunner
            self._backend = LiteralShapeRunner(
                self.source, self.pipeline, self.cache_dir, self.options)

    @property
    def mode(self) -> str:
        return type(self._backend).__name__

    def __call__(self, *tensors):
        return self._backend(*tensors)


__all__ = ["ShapeRunner", "uses_symbolic_shapes"]
