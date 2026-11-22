"""Fixed-shape CUDA Graph replay for prepared execution plans.

The graph removes host guards, allocation and Triton dispatch from the hot path.
It is safe only for a fixed metadata context: static input buffers are copied
into before replay, and the captured output buffers are reused.  For dynamic
shapes use DynamicShapeRunner or re-prepare; this is a deployment primitive, not
a symbolic-shape runtime.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphRunner:
    """Capture one prepared plan and replay it against static buffers.

    ``__call__(*tensors)`` copies the caller's tensors into the static inputs and
    replays. ``replay()`` skips the copies and is appropriate when the caller
    owns the static inputs and updates them in place. The returned outputs alias
    the captured output buffers; copy them before the next replay if needed.
    """

    plan: object
    source_args: tuple

    def __post_init__(self) -> None:
        import torch
        if not self.source_args:
            raise ValueError("GraphRunner requires at least one input")
        if any(not isinstance(a, torch.Tensor) or a.device.type != "cuda" for a in self.source_args):
            raise ValueError("GraphRunner requires CUDA tensors")
        device = self.source_args[0].device
        if any(a.device != device for a in self.source_args):
            raise ValueError("GraphRunner requires one CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("capture GraphRunner outside an existing capture")
        self._torch = torch
        self._device = device
        self._inputs = tuple(torch.empty_like(a) for a in self.source_args)
        for target, source in zip(self._inputs, self.source_args):
            target.copy_(source)
        self._buffers = self.plan.allocate_buffers()
        for _ in range(3):
            self.plan.run_into(*self._inputs, buffers=self._buffers)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._outputs = self.plan.run_into(*self._inputs, buffers=self._buffers)
        self._graph = graph
        self._graph.replay()
        torch.cuda.synchronize(device)

    def _check(self, tensors) -> None:
        if len(tensors) != len(self._inputs):
            raise ValueError(f"GraphRunner expects {len(self._inputs)} inputs")
        for target, value in zip(self._inputs, tensors):
            if (not isinstance(value, self._torch.Tensor)
                    or value.device != self._device
                    or tuple(value.shape) != tuple(target.shape)
                    or value.dtype != target.dtype):
                raise ValueError("GraphRunner input metadata/device changed")

    def copy_in(self, *tensors) -> None:
        self._check(tensors)
        pairs = [(target, value) for target, value in zip(self._inputs, tensors)
                 if value is not target]
        if not pairs:
            return
        if len(pairs) == 1:
            pairs[0][0].copy_(pairs[0][1])
            return
        # One foreach dispatch instead of one Python dispatch per input. The
        # fallback keeps older torch versions working.
        foreach = getattr(self._torch, "_foreach_copy_", None)
        if foreach is None:
            for target, value in pairs:
                target.copy_(value)
            return
        foreach([target for target, _ in pairs], [value for _, value in pairs])

    def replay(self):
        self._graph.replay()
        return self._outputs

    def __call__(self, *tensors):
        self.copy_in(*tensors)
        return self.replay()

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs


def capture_prepared_plan(plan, args) -> GraphRunner:
    """Convenience wrapper matching the ``prepare_execution_plan`` API."""
    return GraphRunner(plan, tuple(args))


__all__ = ["GraphRunner", "capture_prepared_plan"]
