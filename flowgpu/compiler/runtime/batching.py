"""Fixed-capacity dynamic batching for a compiled decode step.

The plan is compiled for a fixed batch size. This helper packs active requests
into that batch, pads empty slots, calls the step, and returns only the
outputs of active requests. Cache blocks and positions remain the caller's
responsibility (see BlockTable / PagedKVCache).
"""
from __future__ import annotations


class DynamicBatch:
    """Track active requests in fixed slots and pad each step."""

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._payloads = {}
        self._slots = {}

    def __len__(self):
        return len(self._payloads)

    @property
    def active(self):
        return dict(self._slots)

    def add(self, request_id, payload=None):
        if request_id in self._payloads:
            raise ValueError("request already active")
        if len(self._payloads) >= self.capacity:
            raise MemoryError("dynamic batch is full")
        free = next(slot for slot in range(self.capacity) if slot not in self._slots.values())
        self._slots[request_id] = free
        self._payloads[request_id] = payload
        return free

    def update(self, request_id, payload):
        if request_id not in self._payloads:
            raise KeyError(f"unknown request: {request_id!r}")
        self._payloads[request_id] = payload

    def remove(self, request_id):
        if request_id not in self._payloads:
            raise KeyError(f"unknown request: {request_id!r}")
        self._slots.pop(request_id)
        self._payloads.pop(request_id)

    def step(self, call, empty):
        """Call ``call(padded_payloads)`` and return outputs for active slots.

        ``empty()`` builds a padded payload for an inactive slot. ``call``
        returns one output per batch slot, in slot order.
        """
        padded = [empty() for _ in range(self.capacity)]
        for request_id, slot in self._slots.items():
            padded[slot] = self._payloads[request_id]
        outputs = call(padded)
        if len(outputs) != self.capacity:
            raise ValueError("step must return one output per batch slot")
        return {request_id: outputs[slot]
                for request_id, slot in self._slots.items()}


__all__ = ["DynamicBatch"]
