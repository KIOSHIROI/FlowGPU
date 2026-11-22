"""conv2d 1x1 is a composition of reshape, transpose, matmul and bias."""
import math

from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.operators import OPERATORS

SOURCE = """
pipeline p {
  input x:fp16[2, 2, 2, 4]
  input weight:fp16[3, 4]
  input bias:fp16[3]
  out = conv2d(x, weight, bias)
  output out
}
"""


def test_conv2d_native_lowering_matches_1x1():
    assert "conv2d" in OPERATORS
    compiled = compile_source(SOURCE)
    pipeline = compiled.pipelines[0]
    inputs = reference.inputs_for(pipeline.reference, {})
    values = [list(entry[3]) for entry in inputs]
    namespace = {}
    exec(compile(reference.module([pipeline.reference]), "<conv_test>", "exec"), namespace)
    actual = namespace["run_p"](*values)["out"]
    x, weight, bias = values
    n, h, w, c = 2, 2, 2, 4
    o = 3
    expected = []
    for ni in range(n):
        for hi in range(h):
            for wi in range(w):
                for oi in range(o):
                    total = bias[oi]
                    for ci in range(c):
                        total += x[((ni * h + hi) * w + wi) * c + ci] * weight[oi * c + ci]
                    expected.append(total)
    assert len(actual) == len(expected)
    assert max(abs(a - b) for a, b in zip(actual, expected)) < 2e-2


THREE_BY_THREE = """
pipeline p {
  input x:fp16[2, 6, 6, 8]
  input weight:fp16[16, 8, 3, 3]
  input bias:fp16[16]
  out = conv2d(x, weight, bias)
  output out
}
"""


def test_conv2d_3x3_native_lowering_matches_reference():
    compiled = compile_source(THREE_BY_THREE)
    pipeline = compiled.pipelines[0]
    values = reference.run(pipeline.reference)
    inputs = reference.inputs_for(pipeline.reference, {})
    x, weight, bias = (item[3] for item in inputs)
    out = values["out"]
    assert len(out) == 2 * 4 * 4 * 16
    # spot-check one output element against an explicit sum
    n, oh, ow, o = 1, 2, 1, 3
    total = 0.0
    for c in range(8):
        for kh in range(3):
            for kw in range(3):
                total += x[((n * 6 + oh + kh) * 6 + ow + kw) * 8 + c] * weight[o * 8 * 9 + c * 9 + kh * 3 + kw]
    total += bias[o]
    index = ((n * 4 + oh) * 4 + ow) * 16 + o
    assert abs(out[index] - total) < 2e-2


SAME = """
pipeline p {
  input x:fp16[2, 6, 6, 8]
  input weight:fp16[16, 8, 3, 3]
  input bias:fp16[16]
  out = conv2d_same(x, weight, bias)
  output out
}
"""


def test_conv2d_same_reference_shape_and_spot_check():
    compiled = compile_source(SAME)
    pipeline = compiled.pipelines[0]
    values = reference.run(pipeline.reference)
    inputs = reference.inputs_for(pipeline.reference, {})
    x, weight, bias = (item[3] for item in inputs)
    out = values["out"]
    assert len(out) == 2 * 6 * 6 * 16
    n, oh, ow, o = 0, 0, 0, 2
    total = 0.0
    for c in range(8):
        for kh in range(3):
            for kw in range(3):
                ih, iw = oh + kh - 1, ow + kw - 1
                if 0 <= ih < 6 and 0 <= iw < 6:
                    total += x[((n * 6 + ih) * 6 + iw) * 8 + c] * weight[o * 8 * 9 + c * 9 + kh * 3 + kw]
    total += bias[o]
    index = ((n * 6 + oh) * 6 + ow) * 16 + o
    assert abs(out[index] - total) < 2e-2
