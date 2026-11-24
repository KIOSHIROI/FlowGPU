"""Continuous batching over the compiled batched decode-step pipeline.

The GPU batch is a fixed literal capacity (the compiled kernel's batch). The
scheduler keeps a slot table, admits requests into free slots, and retires them
when they reach their token budget. Free slots are zeroed on admission so a new
request never sees another request's cache.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestState:
    request_id: int
    position: int = 0
    max_tokens: int = 8
    prompt_length: int = 0
    prompt_k: object = None
    prompt_v: object = None
    finished: bool = False


@dataclass
class ContinuousBatchSession:
    source: str
    pipeline: str
    cache_dir: str
    capacity: int = 4
    heads: int = 2
    keys: int = 128
    depth: int = 64
    query_heads: int = 4
    inplace: bool = False
    slots: list = field(default_factory=list)  # RequestState | None
    _pos: list = field(default_factory=list)
    _tokens: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        import torch
        from flowgpu.compiler.runtime.runtime import compile_pipeline
        self._torch = torch
        self._run = compile_pipeline(self.source, pipeline=self.pipeline,
                                     cache_dir=self.cache_dir)
        self._device = None
        self._k = None
        self._v = None
        self.slots = [None] * self.capacity
        self._pos = [0] * self.capacity

    def _ensure(self, q) -> None:
        import torch
        if self._k is None:
            self._device = q.device
            self._k = torch.zeros(self.capacity, self.heads, self.keys, self.depth,
                                  device=q.device, dtype=q.dtype)
            self._v = torch.zeros_like(self._k)

    def active(self) -> int:
        return sum(slot is not None for slot in self.slots)

    def admit(self, request: RequestState) -> bool:
        for index, slot in enumerate(self.slots):
            if slot is None:
                self.slots[index] = request
                if self._k is not None:
                    self._k[index].zero_()
                    self._v[index].zero_()
                length = request.prompt_length
                if length and self._k is not None:
                    if request.prompt_k is None or request.prompt_v is None:
                        raise ValueError("prompt_length requires prompt_k and prompt_v")
                    self._k[index, :, :length].copy_(request.prompt_k)
                    self._v[index, :, :length].copy_(request.prompt_v)
                request.position = length
                self._pos[index] = length
                return True
        return False

    def _pack(self, tokens):
        import torch
        q = torch.zeros(self.capacity, self.query_heads, 1, self.depth,
                        device=self._device, dtype=self._k.dtype)
        k_new = torch.zeros(self.capacity, self.heads, self.depth,
                            device=self._device, dtype=self._k.dtype)
        v_new = torch.zeros_like(k_new)
        loc = torch.zeros(self.capacity, device=self._device, dtype=torch.int32)
        for index, slot in enumerate(self.slots):
            if slot is None:
                continue
            values = tokens.get(slot.request_id)
            if values is None:
                continue
            q[index], k_new[index], v_new[index] = values
            loc[index] = self._pos[index]
        return q, k_new, v_new, loc

    def _run_inplace(self, tokens) -> dict:
        q = self._torch.zeros(self.capacity, self.query_heads, 1, self.depth,
                              device=self._device, dtype=self._k.dtype)
        for index, slot in enumerate(self.slots):
            if slot is None:
                continue
            values = tokens.get(slot.request_id)
            if values is None:
                continue
            q[index], k_row, v_row = values
            self._k[index, :, self._pos[index]].copy_(k_row)
            self._v[index, :, self._pos[index]].copy_(v_row)
        return self._run(q, self._k, self._v)

    def step(self, tokens) -> dict:
        self._ensure(next(iter(tokens.values()))[0] if tokens else
                     self._torch.zeros(self.capacity, self.query_heads, 1, self.depth,
                                       device="cuda", dtype=self._torch.bfloat16))
        if self.inplace:
            result = self._run_inplace(tokens)
        else:
            q, k_new, v_new, loc = self._pack(tokens)
            result = self._run(q, self._k, self._v, k_new, v_new, loc)
            self._k, self._v = result["k2"], result["v2"]
        outputs = {}
        for index, slot in enumerate(self.slots):
            if slot is None:
                continue
            outputs[slot.request_id] = result["out"][index]
            slot.position += 1
            self._pos[index] = slot.position
            self._tokens += 1
            if slot.position >= slot.max_tokens:
                slot.finished = True
                self.slots[index] = None
        return outputs

    @property
    def generated_tokens(self) -> int:
        return self._tokens


__all__ = ["RequestState", "ContinuousBatchSession"]
