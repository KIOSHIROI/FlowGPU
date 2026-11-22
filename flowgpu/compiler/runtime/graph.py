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
    copy_inputs: bool = True

    def __post_init__(self) -> None:
        import torch
        if not self.source_args:
            raise ValueError("GraphRunner requires at least one input")
        tensors = [a for a in self.source_args if isinstance(a, torch.Tensor)]
        scalars = [a for a in self.source_args if not isinstance(a, torch.Tensor)]
        if not tensors or any(a.device.type != "cuda" for a in tensors):
            raise ValueError("GraphRunner requires CUDA tensors")
        if any(type(a) not in {int, float} for a in scalars):
            raise ValueError("GraphRunner requires tensor or Python scalar inputs")
        device = tensors[0].device
        if any(a.device != device for a in tensors):
            raise ValueError("GraphRunner requires one CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("capture GraphRunner outside an existing capture")
        self._torch = torch
        self._device = device
        if self.copy_inputs:
            self._inputs = tuple(torch.empty_like(a) if isinstance(a, torch.Tensor) else a
                                 for a in self.source_args)
            for target, source in zip(self._inputs, self.source_args):
                if isinstance(target, torch.Tensor):
                    target.copy_(source)
        else:
            # Caller-owned static buffers: update them in place between replays.
            self._inputs = tuple(self.source_args)
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
            if isinstance(target, self._torch.Tensor):
                if (not isinstance(value, self._torch.Tensor)
                        or value.device != self._device
                        or tuple(value.shape) != tuple(target.shape)
                        or value.dtype != target.dtype):
                    raise ValueError("GraphRunner input metadata/device changed")
            elif type(value) not in {int, float} or value != target:
                raise ValueError("GraphRunner scalar input changed; capture a new graph")

    def copy_in(self, *tensors) -> None:
        if not self.copy_inputs:
            raise ValueError("GraphRunner was created with copy_inputs=False; "
                             "update runner.inputs in place and call replay()")
        self._check(tensors)
        pairs = [(target, value) for target, value in zip(self._inputs, tensors)
                 if isinstance(target, self._torch.Tensor) and value is not target]
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
        if not self.copy_inputs:
            raise ValueError("GraphRunner was created with copy_inputs=False; "
                             "call replay() instead")
        self.copy_in(*tensors)
        return self.replay()

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs


def capture_prepared_plan(plan, args, *, copy_inputs: bool = True) -> GraphRunner:
    """Convenience wrapper matching the ``prepare_execution_plan`` API."""
    return GraphRunner(plan, tuple(args), copy_inputs=copy_inputs)


def prepare_graph_pipeline(source, *, pipeline, args, cache_dir, validator,
                           copy_inputs: bool = True, **options) -> GraphRunner:
    """Prepare an execution plan and capture it into a ready GraphRunner.

    With the default ``copy_inputs=True``, calls accept new tensors (metadata
    must match) and copy them into the captured static inputs. With
    ``copy_inputs=False``, callers must own and update ``runner.inputs``.
    """
    from flowgpu.compiler.runtime.execution_plan import prepare_execution_plan
    plan = prepare_execution_plan(source, pipeline=pipeline, args=args,
                                  cache_dir=cache_dir, validator=validator, **options)
    return GraphRunner(plan, tuple(args), copy_inputs=copy_inputs)


__all__ = ["GraphRunner", "capture_prepared_plan", "prepare_graph_pipeline"]
