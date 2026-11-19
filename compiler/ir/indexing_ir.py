"""Explicit read-index and scatter-write contracts.

These contracts are preparatory IR only. The source language does not expose
general indexing or scatter yet, and no backend consumes them automatically.
Making the alias and conflict policy explicit prevents a writable view from
being inferred from a read-only packed-KV slice.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexSpec:
    source: str
    index_values: str
    axis: int
    result_shape: tuple
    masked: bool = False
    bounds_check: str = "reject"

    def __post_init__(self):
        object.__setattr__(self, "result_shape", tuple(str(x) for x in self.result_shape))
        if not self.source or not self.index_values:
            raise ValueError("index spec needs source and index values")
        if type(self.axis) is not int or self.axis < 0:
            raise ValueError("index axis must be a non-negative static integer")
        if not self.result_shape:
            raise ValueError("index result shape cannot be empty")
        if self.bounds_check not in {"reject", "mask", "clamp"}:
            raise ValueError("invalid index bounds policy")
        if self.bounds_check == "mask" and not self.masked:
            raise ValueError("mask bounds policy requires masked indexing")


@dataclass(frozen=True)
class ScatterWriteSpec:
    target: str
    index: IndexSpec
    values: str
    conflict: str = "reject"
    atomic: bool = False
    aliasing: str = "no_alias"

    def __post_init__(self):
        if not self.target or not self.values:
            raise ValueError("scatter write needs target and values")
        if self.index.source != self.target:
            raise ValueError("scatter index source must be the scatter target")
        if self.conflict not in {"reject", "last", "add", "max"}:
            raise ValueError("invalid scatter conflict policy")
        if self.conflict in {"add", "max"} and not self.atomic:
            raise ValueError("reduction scatter conflicts require atomic=True")
        if self.aliasing not in {"no_alias", "may_alias"}:
            raise ValueError("invalid scatter aliasing policy")
        if self.aliasing == "may_alias" and self.conflict == "reject":
            raise ValueError("may-alias scatter needs an explicit conflict policy")

    def as_dict(self):
        return {
            "target": self.target,
            "index": {
                "source": self.index.source,
                "index_values": self.index.index_values,
                "axis": self.index.axis,
                "result_shape": list(self.index.result_shape),
                "masked": self.index.masked,
                "bounds_check": self.index.bounds_check,
            },
            "values": self.values,
            "conflict": self.conflict,
            "atomic": self.atomic,
            "aliasing": self.aliasing,
        }
