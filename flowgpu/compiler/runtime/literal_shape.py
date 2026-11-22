"""Per-shape specialization for programs that declare literal input extents.

A literal program cannot simply be recompiled with ``shape_bindings``: the
extent is part of the declaration.  This runner rewrites only the named input
declarations for each observed shape and compiles that static source. It is a
fallback for programs with no symbolic form (simple pointwise programs); a
literal in the body that depends on the input extent may make the rewrite fail,
and that failure is explicit rather than silently wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def rewrite_input_shapes(source: str, shapes: dict[str, tuple[int, ...]]) -> str:
    """Replace the bracket extents of named ``input`` declarations only."""
    rewritten = source
    for name, shape in shapes.items():
        pattern = re.compile(
            rf"(input\s+{re.escape(name)}\s*:\s*[A-Za-z_][A-Za-z0-9_]*\s*\[)([^\]]*)(\])")
        replacement = ",".join(str(int(extent)) for extent in shape)
        rewritten, count = pattern.subn(lambda match: match.group(1) + replacement + match.group(3),
                                        rewritten, count=1)
        if count != 1:
            raise ValueError(f"could not rewrite the declaration of input {name!r}")
    return rewritten


@dataclass
class LiteralShapeRunner:
    source: str
    pipeline: str
    cache_dir: str
    options: dict | None = None
    _runs: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        from flowgpu.compiler.compile import compile_source
        from flowgpu.compiler.runtime.runtime import compile_pipeline
        self.options = dict(self.options or {})
        if "shape_bindings" in self.options:
            raise ValueError("LiteralShapeRunner rewrites literal input declarations")
        compiled = compile_source(self.source, emit=False, **self.options)
        selected = None
        for item in compiled.pipelines:
            if item.pipeline.name == self.pipeline:
                selected = item
                break
        if selected is None:
            raise ValueError(f"no pipeline {self.pipeline!r}")
        self._names = tuple(statement.name for statement in selected.pipeline.statements
                            if getattr(statement, "type", None) is not None
                            and type(statement).__name__ == "InputDecl")
        self._base_run = compile_pipeline(
            self.source, pipeline=self.pipeline, cache_dir=self.cache_dir,
            **self.options)

    def _run_for(self, tensors):
        key = tuple(tuple(int(extent) for extent in tensor.shape) for tensor in tensors)
        if key in self._runs:
            return self._runs[key]
        shapes = dict(zip(self._names, key))
        rewritten = rewrite_input_shapes(self.source, shapes)
        from flowgpu.compiler.runtime.runtime import compile_pipeline
        run = compile_pipeline(rewritten, pipeline=self.pipeline,
                               cache_dir=self.cache_dir, **self.options)
        self._runs[key] = run
        return run

    def __call__(self, *tensors):
        if len(tensors) != len(self._names):
            raise ValueError(f"expected {len(self._names)} input tensors, got {len(tensors)}")
        key = tuple(tuple(int(extent) for extent in tensor.shape) for tensor in tensors)
        # The base module is valid only for the declared shapes; cache it under
        # that key and rewrite the declarations for every other shape.
        if not self._runs:
            self._runs[key] = self._base_run
        return self._run_for(tensors)(*tensors)


__all__ = ["LiteralShapeRunner", "rewrite_input_shapes"]
