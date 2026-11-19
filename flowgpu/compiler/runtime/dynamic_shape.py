"""Per-shape dispatch for symbolic FlowGPU programs.

FlowGPU does not compile a symbolic-shape kernel.  This module makes the
practical alternative explicit and reusable: derive the binding values from the
actual tensors, compile one static specialization per distinct metadata tuple,
and cache it through ``ShapeSpecializedRunner``.

It is not a dynamic-shape compiler.  A new shape is a compile miss; the guard is
the complete tensor metadata (shape, stride, dtype, device, alignment).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.ir import INPUT, DataflowGraph
from flowgpu.compiler.runtime.runtime import compile_pipeline
from flowgpu.compiler.runtime.shape_specialization import ShapeSpecializedRunner


def derive_bindings(graph: DataflowGraph, tensors) -> dict[str, int]:
    """Map the graph's symbolic dimensions to concrete tensor extents.

    Literal extents are validated rather than silently accepted.  A symbolic
    name that appears twice must receive the same value; a mismatch is a caller
    error, not a new specialization.
    """
    inputs = [node for node in graph.nodes if node.op == INPUT]
    if len(inputs) != len(tensors):
        raise ValueError(f"expected {len(inputs)} input tensors, got {len(tensors)}")
    bindings: dict[str, int] = {}
    for node, tensor in zip(inputs, tensors):
        declared = graph.values[node.output]
        actual = tuple(int(value) for value in tensor.shape)
        if len(declared.shape) != len(actual):
            raise ValueError(
                f"input {node.output!r} expects rank {len(declared.shape)}, got {len(actual)}")
        for declared_extent, extent in zip(declared.shape, actual):
            text = str(declared_extent)
            if text.isdigit():
                if int(text) != extent:
                    raise ValueError(
                        f"input {node.output!r} expects extent {text}, got {extent}")
                continue
            previous = bindings.get(text)
            if previous is not None and previous != extent:
                raise ValueError(
                    f"symbol {text!r} binds to both {previous} and {extent}")
            bindings[text] = extent
    return bindings


def _pipeline(compiled, name: str):
    for pipeline in compiled.pipelines:
        if pipeline.pipeline.name == name:
            return pipeline
    raise ValueError(f"no pipeline {name!r}; found "
                     f"{[p.pipeline.name for p in compiled.pipelines]}")


@dataclass
class DynamicShapeRunner:
    """Compile and cache one static specialization per observed input metadata.

    ``build`` options are passed to ``compile_pipeline``; ``shape_bindings`` is
    derived per call and may not be supplied by the caller.  The base graph is
    compiled once with ``emit=False`` to learn the symbolic dimensions.
    """

    source: str
    pipeline: str
    cache_dir: str | Path
    options: dict | None = None
    max_entries: int = 16

    def __post_init__(self) -> None:
        self.options = dict(self.options or {})
        if "shape_bindings" in self.options:
            raise ValueError("DynamicShapeRunner derives shape_bindings per call")
        compiled = compile_source(self.source, emit=False, **self.options)
        self._graph = _pipeline(compiled, self.pipeline).dataflow
        self._specializer = ShapeSpecializedRunner(self._build, max_entries=self.max_entries)

    def _build(self, *args):
        bindings = derive_bindings(self._graph, args)
        return compile_pipeline(
            self.source,
            pipeline=self.pipeline,
            cache_dir=str(self.cache_dir),
            shape_bindings=bindings,
            **self.options,
        )

    def __call__(self, *tensors):
        return self._specializer(*tensors)

    @property
    def hits(self) -> int:
        return self._specializer.hits

    @property
    def misses(self) -> int:
        return self._specializer.misses

    def __len__(self) -> int:
        return len(self._specializer)


__all__ = ["DynamicShapeRunner", "derive_bindings"]
