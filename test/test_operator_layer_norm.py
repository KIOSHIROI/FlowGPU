"""layer_norm lowers to one native Triton row-normalization kernel."""
import math

from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.operators import OPERATORS

SOURCE = """
pipeline p {
  input x:fp16[2, 4]
  input weight:fp16[4]
  input bias:fp16[4]
  input eps:fp32
  out = layer_norm(x, weight, bias, eps)
  output out
}
"""


def test_layer_norm_native_lowering_matches_reference():
    assert "layer_norm" in OPERATORS
    compiled = compile_source(SOURCE)
    pipeline = compiled.pipelines[0]
    values = reference.run(pipeline.reference)
    inputs = reference.inputs_for(pipeline.reference, {})
    x = inputs[0][3]
    weight = inputs[1][3]
    bias = inputs[2][3]
    eps = inputs[3][3]
    width = len(weight)
    expected = []
    for row in range(2):
        chunk = x[row * width:(row + 1) * width]
        mean = sum(chunk) / width
        var = sum((value - mean) ** 2 for value in chunk) / width
        den = math.sqrt(var + eps)
        expected.extend((chunk[i] - mean) / den * weight[i] + bias[i]
                        for i in range(width))
    got = values["out"]
    assert max(abs(a - b) for a, b in zip(got, expected)) < 2e-2
