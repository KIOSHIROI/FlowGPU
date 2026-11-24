"""Host-side reuse of one prepared subgraph across layers with different weights.

A decoder has many structurally equal subgraphs with distinct weight tensors.
The compiled plan is shape-guarded, so one plan can serve every layer; only the
weight arguments change. Alternating output buffer sets lets the previous
layer's output be the next layer's input without an alias or a copy.
"""
from __future__ import annotations


class LayerStack:
    """Run one prepared plan across a sequence of per-layer argument tuples."""

    def __init__(self, plan, layers, *, output_name="y"):
        self._plan = plan
        self._layers = tuple(tuple(layer) for layer in layers)
        self._output_name = output_name
        # Two output buffer sets: layer i writes set i % 2, layer i+1 reads it
        # as input while writing the other set. No overlap, no copy.
        self._buffers = (plan.allocate_buffers(), plan.allocate_buffers())

    @property
    def layers(self):
        return self._layers

    def __call__(self, value):
        current = value
        target = 0
        for layer in self._layers:
            result = self._plan.run_into(current, *layer,
                                         buffers=self._buffers[target])
            current = result[self._output_name]
            target ^= 1
        return current


__all__ = ["LayerStack"]
