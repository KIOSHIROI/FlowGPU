"""Restricted zero-copy tensor views used by packed-KV lowering candidates.

This is intentionally an IR contract, not general indexing syntax. A view has
static rank, explicit element strides and an offset; it is read-only and may
only be consumed by a backend that accepts its contiguity contract.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewSpec:
    source: str
    shape: tuple
    strides: tuple
    offset: int = 0
    memory_scope: str = "global"
    read_only: bool = True
    contiguous: bool = False

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(str(x) for x in self.shape))
        object.__setattr__(self, "strides", tuple(str(x) for x in self.strides))
        if not self.source:
            raise ValueError("view needs a source value")
        if not self.shape or len(self.shape) != len(self.strides):
            raise ValueError("view shape and strides must have the same non-zero rank")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("view offset must be a non-negative element offset")
        if self.memory_scope not in {"global", "shared", "local", "register"}:
            raise ValueError("invalid view memory scope")
        if not self.read_only:
            raise ValueError("restricted views are read-only")


@dataclass(frozen=True)
class PackedKVSlice:
    source: str
    axis: int
    start: int
    stop: int
    value: ViewSpec

    def __post_init__(self):
        rank = len(self.value.shape)
        if type(self.axis) is not int or not -rank <= self.axis < rank:
            raise ValueError("packed-KV slice axis is out of range")
        axis = self.axis % rank
        object.__setattr__(self, "axis", axis)
        if self.value.source != self.source:
            raise ValueError("packed-KV view source does not match slice source")
        if type(self.start) is not int or type(self.stop) is not int:
            raise ValueError("packed-KV slice bounds must be static integers")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("packed-KV slice must have positive static width")
        width = self.value.shape[axis]
        if width.isdigit() and self.stop > int(width):
            raise ValueError("packed-KV slice exceeds the source axis")

    @property
    def width(self):
        return self.stop - self.start

    def as_dict(self):
        return {
            "source": self.source,
            "axis": self.axis,
            "start": self.start,
            "stop": self.stop,
            "width": self.width,
            "shape": list(self.value.shape),
            "strides": list(self.value.strides),
            "offset": self.value.offset,
            "contiguous": self.value.contiguous,
            "read_only": self.value.read_only,
        }
