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
    prepared: bool = False
    inplace: bool = False
    graph: bool = False
    _pos: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        import torch
        from flowgpu.compiler.runtime.runtime import compile_pipeline
        self._torch = torch
        if self.graph and not self.inplace:
            raise ValueError("graph decode requires inplace=True")
        self._runner = None
        self._q_static = None
        if self.prepared:
            self._plan = None
            self._buffers = None
            self._run = None
        else:
            self._plan = None
            self._run = compile_pipeline(self.source, pipeline=self.pipeline,
                                         cache_dir=self.cache_dir)
        self._device = None
        self._k = None
        self._v = None

    def reset(self) -> None:
        self._pos = 0
        # Reallocate: in prepared mode the old buffers belong to a plan output
        # set and must not alias the next call's inputs.
        self._k = None
        self._v = None
        self._runner = None
        self._q_static = None
        self._next_buffer = 0

    def seed(self, k_cache, v_cache, position: int) -> None:
        """Install a prefill-produced cache before decoding."""
        self._k = k_cache
        self._v = v_cache
        self._pos = position
        self._device = k_cache.device
        self._runner = None
        self._q_static = None

    def _ensure(self, q):
        import torch
        if self._k is None:
            self._device = q.device
            self._k = torch.zeros(self.heads, self.keys, self.depth,
                                  device=q.device, dtype=q.dtype)
            self._v = torch.zeros_like(self._k)
        if self._pos >= self.keys:
            raise ValueError("decode session cache is full; reset before continuing")

    def _capture_graph(self, q) -> None:
        from flowgpu.compiler.runtime.execution_plan import prepare_execution_plan
        from flowgpu.compiler.runtime.graph import GraphRunner
        self._q_static = self._torch.empty_like(q)
        plan = prepare_execution_plan(
            self.source, pipeline=self.pipeline,
            args=(self._q_static, self._k, self._v),
            cache_dir=self.cache_dir, validator=lambda result: True)
        self._runner = GraphRunner(plan, (self._q_static, self._k, self._v),
                                   copy_inputs=False)

    def step(self, q, k_new, v_new):
        self._ensure(q)
        if self.graph:
            if self._runner is None:
                self._capture_graph(q)
            self._k[:, self._pos].copy_(k_new)
            self._v[:, self._pos].copy_(v_new)
            self._q_static.copy_(q)
            result = self._runner.replay()
            self._pos += 1
            return result["out"]
        if self.inplace:
            self._k[:, self._pos].copy_(k_new)
            self._v[:, self._pos].copy_(v_new)
            if self._plan is not None:
                result = self._plan.run_into(q, self._k, self._v,
                                             buffers=self._buffers[0])
            else:
                result = self._run(q, self._k, self._v)
            self._pos += 1
            return result["out"]
        loc = self._torch.tensor([self._pos], device=q.device, dtype=self._torch.int32)
        if self.prepared and self._plan is None:
            from flowgpu.compiler.runtime.execution_plan import prepare_execution_plan
            self._plan = prepare_execution_plan(
                self.source, pipeline=self.pipeline,
                args=(q, self._k, self._v, k_new, v_new, loc),
                cache_dir=self.cache_dir, validator=lambda result: True)
            self._buffers = [self._plan.allocate_buffers(),
                             self._plan.allocate_buffers()]
            self._next_buffer = 0
        if self._plan is not None:
            target = self._buffers[self._next_buffer]
            self._next_buffer ^= 1
            result = self._plan.run_into(q, self._k, self._v, k_new, v_new, loc,
                                         buffers=target)
        else:
            result = self._run(q, self._k, self._v, k_new, v_new, loc)
        self._k, self._v = result["k2"], result["v2"]
        self._pos += 1
        return result["out"]

    @property
    def position(self) -> int:
        return self._pos


__all__ = ["DecodeSession"]
