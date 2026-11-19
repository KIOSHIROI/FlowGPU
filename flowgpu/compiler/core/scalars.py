"""Typed compile-time scalar contracts.

The first scalar phase intentionally supports only static integer parameters.
These values are suitable for axes, slice bounds, and future TopK parameters;
they are not runtime tensors and do not participate in implicit broadcasting.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ScalarType:
    dtype: str
    static: bool = True

    def __post_init__(self):
        if self.dtype not in {"int32", "int64", "fp32"}:
            raise ValueError(f"unsupported scalar dtype: {self.dtype}")
        if type(self.static) is not bool:
            raise TypeError("scalar static flag must be bool")
        if not self.static:
            raise ValueError("runtime scalar types are not enabled yet")


@dataclass(frozen=True)
class TypedScalar:
    value: int | float
    type: ScalarType

    def __post_init__(self):
        if self.type.dtype in {"int32", "int64"} and type(self.value) is not int:
            raise TypeError("integer scalar needs a Python int value")
        if self.type.dtype == "fp32" and type(self.value) not in {int, float}:
            raise TypeError("fp32 scalar needs a numeric value")


def static_int(value, *, dtype="int32"):
    if type(value) is not int:
        raise TypeError("static integer scalar needs an int")
    return TypedScalar(value, ScalarType(dtype))
