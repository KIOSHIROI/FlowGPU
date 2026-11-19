"""Per-shape specialization without a dynamic-shape compiler.

FlowGPU still compiles static shapes.  A deployment that sees several shapes can
nevertheless cache one compiled/prepared runner per input metadata tuple: shape,
stride, dtype, device and pointer alignment.  The first call for a tuple builds
the runner; later calls with the same tuple hit the cache.  A different tuple is
a miss, never a silent reuse.

This is a guard around explicit specialization, not symbolic shape inference.
The builder is supplied by the caller, so it can compile new source, prepare a
frozen autotune module, or return an already declared function.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    device: str
    alignment_mod128: int | None = None

    def as_dict(self) -> dict:
        return {
            "shape": list(self.shape),
            "stride": list(self.stride),
            "dtype": self.dtype,
            "device": self.device,
            "alignment_mod128": self.alignment_mod128,
        }


def tensor_metadata(tensor, *, alignment: bool = True) -> TensorMetadata:
    """Read the metadata a compiled kernel's guard depends on."""
    stride = tensor.stride() if callable(getattr(tensor, "stride", None)) else tensor.stride
    dtype = getattr(tensor, "dtype", None)
    device = getattr(tensor, "device", None)
    align = None
    if alignment and callable(getattr(tensor, "data_ptr", None)):
        align = int(tensor.data_ptr()) % 128
    return TensorMetadata(
        shape=tuple(int(v) for v in tensor.shape),
        stride=tuple(int(v) for v in stride),
        dtype=str(dtype),
        device=str(device),
        alignment_mod128=align,
    )


class ShapeSpecializedRunner:
    """Cache one built runner per complete input metadata tuple.

    ``build(*args)`` must return a callable.  Exceptions propagate; a failed
    build is not cached.  The cache is bounded and evicts the least-recently
    used tuple, so a workload with unbounded shapes cannot exhaust memory.
    """

    def __init__(self, build: Callable, *, max_entries: int = 16,
                 alignment: bool = True) -> None:
        if not callable(build):
            raise TypeError("build must be callable")
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._build = build
        self._alignment = alignment
        self._max_entries = max_entries
        self._runners: OrderedDict[tuple, Callable] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def key(self, args) -> tuple:
        return tuple(tensor_metadata(arg, alignment=self._alignment) for arg in args)

    def __call__(self, *args):
        key = self.key(args)
        runner = self._runners.get(key)
        if runner is not None:
            self.hits += 1
            self._runners.move_to_end(key)
            return runner(*args)
        self.misses += 1
        runner = self._build(*args)
        if not callable(runner):
            raise TypeError("build must return a callable")
        self._runners[key] = runner
        while len(self._runners) > self._max_entries:
            self._runners.popitem(last=False)
        return runner(*args)

    def clear(self) -> None:
        self._runners.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._runners)


__all__ = ["TensorMetadata", "tensor_metadata", "ShapeSpecializedRunner"]
