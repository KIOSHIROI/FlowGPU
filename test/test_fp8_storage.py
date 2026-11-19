"""The fp8 storage dtypes (e4m3 and e5m2): they store, they do not compute.

The FP8 competition operators produce an e4m3 tensor and the quantisers may
produce either format, so the language needs both dtypes -- but 8-bit arithmetic
is not what the language is for.  The class rule says so: `storage` is neither a
float nor an integer, a cast is the only way in or out, and an arithmetic
operator refuses it by name.
"""

import pytest

from compiler.compile import compile_source
from compiler.diagnostic import CompileError


def compile_one(source):
    return compile_source(source).pipelines[0]


@pytest.mark.parametrize("dtype", ["fp8e4m3", "fp8e5m2"])
def test_fp8_is_a_one_byte_storage_dtype(dtype):
    from compiler.analysis.precision import DTYPE_BYTES, DTYPE_CLASSES, STORAGE_DTYPES

    assert DTYPE_BYTES[dtype] == 1
    assert DTYPE_CLASSES[dtype] == "storage"
    assert dtype in STORAGE_DTYPES


@pytest.mark.parametrize("source", [
    "pipeline p { input x : fp32[N] y = cast_to(x, fp8e4m3) output y }",
    "pipeline p { input x : fp8e4m3[N] y = cast_to(x, fp32) output y }",
    "pipeline p { input x : fp16[N] y = cast_to(x, fp8e4m3) output y }",
    "pipeline p { input x : fp32[N] y = cast_to(x, fp8e5m2) output y }",
    "pipeline p { input x : fp8e5m2[N] y = cast_to(x, fp32) output y }",
    "pipeline p { input x : bf16[N] y = cast_to(x, fp8e5m2) output y }",
])
def test_casts_move_values_in_and_out_of_fp8(source):
    assert compile_one(source).triton.kernels


@pytest.mark.parametrize("dtype", ["fp8e4m3", "fp8e5m2"])
@pytest.mark.parametrize("body", ["z = add(x, y)", "z = mul(x, y)", "z = sub(x, y)"])
def test_fp8_arithmetic_is_refused_by_class(body, dtype):
    source = (f"pipeline p {{ input x : {dtype}[N] input y : {dtype}[N] "
              + body + " output z }")
    with pytest.raises(CompileError, match="storage dtype"):
        compile_one(source)


def test_reference_rounds_to_e4m3_nearest_even_and_saturates():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ref_under_test", root / "compiler/backend/reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The rounding helpers live in the emitted reference's preamble, which is the
    # source text the generated module carries, so they are executed rather than
    # imported.
    preamble = {}
    exec(module.PREAMBLE, preamble)

    cases = {0.0: 0.0, 1.0: 1.0, 1.0625: 1.0, 1.1875: 1.25, -1.1875: -1.25,
             0.001953125: 0.001953125, 448.0: 448.0, 1000.0: 448.0,
             -1000.0: -448.0, 2.0 ** -9: 2.0 ** -9}
    for value, expected in cases.items():
        assert preamble['fp8e4m3'](value) == pytest.approx(expected), value


def test_reference_rounds_to_e5m2_with_two_mantissa_bits():
    """e5m2 is IEEE-like: 5 exponent bits, 2 mantissa bits, max finite 57344.

    The expected values are the format's own grid, not a scaled copy of e4m3:
    `1.1875` rounds to `1.25` because there are only two mantissa bits, and the
    smallest normal is `2**-14` with subnormals down to `2**-16`.  Above the
    range e5m2 has infinities, so `61440` (halfway to the next encoding, which
    does not exist) rounds to infinity rather than saturating at 57344.
    """
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "ref_under_test_e5", root / "compiler/backend/reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preamble = {}
    exec(module.PREAMBLE, preamble)
    round_to_e5m2 = preamble['fp8e5']

    cases = {0.0: 0.0, 1.0: 1.0, 1.0625: 1.0, 1.1875: 1.25, -1.1875: -1.25,
             0.001: 0.0009765625, 57343.0: 57344.0, 57344.0: 57344.0,
             2.0 ** -14: 2.0 ** -14, 2.0 ** -17: 2.0 ** -16, 2.0 ** -18: 0.0,
             61440.0: float("inf"), -1e9: float("-inf")}
    for value, expected in cases.items():
        assert round_to_e5m2(value) == pytest.approx(expected), value


def test_storage_arithmetic_error_names_the_repair():
    """T12: a refusal should say how to fix it, not only that it is refused."""
    with pytest.raises(CompileError) as info:
        compile_one("pipeline p { input x : fp8e4m3[N] y = add(x, x) output y }")
    message = str(info.value)
    assert "storage dtype" in message
    assert "cast_to(x, fp32)" in message
