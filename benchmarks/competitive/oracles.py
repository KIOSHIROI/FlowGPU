"""Small independent CPU oracles for the competitive probe.

These are deliberately written from the operator contract rather than from the
FlowGPU reference backend.  They operate on flat Python lists, which is the
shape `backend.reference.inputs_for` produces.
"""
from __future__ import annotations

import math


def _add(values, sizes):
    a, b = values
    return {"out": [x + y for x, y in zip(a, b)]}


def _int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def _neg(values, sizes):
    (x,) = values
    return {"out": [-value for value in x]}


def _pow(values, sizes):
    x, exponent = values
    out = []
    for a, b in zip(x, exponent):
        try:
            out.append(math.pow(a, b))
        except ValueError:
            out.append(math.nan)
        except OverflowError:
            out.append(math.inf)
    return {"out": out}


def _int_pointwise(values, sizes):
    x, y = values
    return {"z": [_int32(_int32(a + b) * c) for a, b, c in zip(x, y, y)]}


def _matmul(values, sizes):
    a, b = values
    m, k, n = sizes["M"], sizes["K"], sizes["N"]
    out = []
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i * k + p] * b[p * n + j]
            out.append(total)
    return {"out": out}


def _softmax(values, sizes):
    (x,) = values
    rows, width = sizes["B"], sizes["V"]
    out = []
    for row in range(rows):
        chunk = x[row * width:(row + 1) * width]
        peak = max(chunk)
        exps = [math.exp(value - peak) for value in chunk]
        total = sum(exps)
        out.extend(value / total for value in exps)
    return {"q": out}


def _residual_norm(values, sizes):
    x, residual, weight = values
    y = [a + b for a, b in zip(x, residual)]
    # rms_norm is over each row; the probe cases use width D.
    width = len(weight)
    out = []
    for start in range(0, len(y), width):
        row = y[start:start + width]
        mean = sum(value * value for value in row) / width
        scale = math.sqrt(mean + 1e-6)
        out.extend(row[i] / scale * weight[i] for i in range(width))
    return {"z": out}


ORACLES = {
    "add": _add,
    "neg": _neg,
    "pow": _pow,
    "int_pointwise": _int_pointwise,
    "matmul": _matmul,
    "softmax": _softmax,
    "residual_norm": _residual_norm,
}


def compare(name, expected, actual, *, atol=1e-4, rtol=1e-4):
    """Return (passed, max_abs_error) for one output list."""
    if name not in expected:
        raise KeyError("oracle has no output " + name)
    want, got = expected[name], actual
    if len(want) != len(got):
        return False, float("inf")
    if not want:
        return True, 0.0
    errors = []
    for a, b in zip(want, got):
        if isinstance(a, int) and not isinstance(a, bool):
            errors.append(0.0 if int(b) == a else float("inf"))
        else:
            errors.append(abs(float(b) - float(a)))
    return all(error <= atol + rtol * abs(float(a)) for error, a in zip(errors, want)), max(errors)
