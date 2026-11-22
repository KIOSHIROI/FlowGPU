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


def test_conv2d_is_composed_and_reference_matches_1x1():
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
    assert any(decision.reason.startswith("conv2d") for decision in pipeline.normalization)
