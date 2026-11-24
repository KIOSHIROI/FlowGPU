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
        self.options = dict(self.options or {})
        if "shape_bindings" in self.options:
            raise ValueError("LiteralShapeRunner rewrites literal input declarations")
        # Parse declarations without lowering: an operator that rejects
        # symbolic extents must still be usable after the per-shape rewrite.
        self._names = tuple(re.findall(r"input\s+([A-Za-z_][A-Za-z0-9_]*)\s*:",
                                       self.source))
        if not self._names:
            raise ValueError("LiteralShapeRunner found no input declarations")
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
        return self._run_for(tensors)(*tensors)


__all__ = ["LiteralShapeRunner", "rewrite_input_shapes"]
