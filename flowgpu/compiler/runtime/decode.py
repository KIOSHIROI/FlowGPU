"""A minimal token-generation session over the compiled decode-step pipeline.

The session owns the KV cache tensors; every step writes one new K/V row and
returns the attention output. It is a runtime convenience, not a second
compiler: the pipeline is still the compiled FlowGPU module.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DecodeSession:
    source: str
    pipeline: str
    cache_dir: str
    heads: int = 2
    keys: int = 128
    depth: int = 64
    query_heads: int = 4
    _pos: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        import torch
        from flowgpu.compiler.runtime.runtime import compile_pipeline
        self._torch = torch
        self._run = compile_pipeline(self.source, pipeline=self.pipeline,
                                     cache_dir=self.cache_dir)
        self._device = None
        self._k = None
        self._v = None

    def reset(self) -> None:
        self._pos = 0
        if self._k is not None:
            self._k.zero_()
            self._v.zero_()

    def _ensure(self, q):
        import torch
        if self._k is None:
            self._device = q.device
            self._k = torch.zeros(self.heads, self.keys, self.depth,
                                  device=q.device, dtype=q.dtype)
            self._v = torch.zeros_like(self._k)
        if self._pos >= self.keys:
            raise ValueError("decode session cache is full; reset before continuing")

    def step(self, q, k_new, v_new):
        self._ensure(q)
        loc = self._torch.tensor([self._pos], device=q.device, dtype=self._torch.int32)
        result = self._run(q, self._k, self._v, k_new, v_new, loc)
        self._k, self._v = result["k2"], result["v2"]
        self._pos += 1
        return result["out"]

    @property
    def position(self) -> int:
        return self._pos


__all__ = ["DecodeSession"]
