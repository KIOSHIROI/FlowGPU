"""batch_norm inference is a composed affine normalization over provided stats."""
import math

from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.operators import OPERATORS

SOURCE = """
pipeline p {
  input x:fp16[2, 4]
  input mean:fp16[4]
  input var:fp16[4]
  input weight:fp16[4]
  input bias:fp16[4]
  input eps:fp32
  out = batch_norm(x, mean, var, weight, bias, eps)
  output out
}
"""


def test_batch_norm_is_composed_and_reference_matches_affine_formula():
    assert "batch_norm" in OPERATORS
    compiled = compile_source(SOURCE)
    pipeline = compiled.pipelines[0]
    program = pipeline.reference
    inputs = reference.inputs_for(program, {})
    values = [list(entry[3]) if entry[1] else entry[3] for entry in inputs]
    values[2] = [abs(value) + 0.5 for value in values[2]]  # variance must be positive
    values[5] = 1e-3
    namespace = {}
    exec(compile(reference.module([program]), "<batch_norm_test>", "exec"), namespace)
    actual = namespace["run_p"](*values)
    x, mean, var, weight, bias, eps = values
    width = len(weight)
    expected = []
    for row in range(2):
        for col in range(width):
            value = x[row * width + col]
            expected.append((value - mean[col]) / math.sqrt(var[col] + eps)
                            * weight[col] + bias[col])
    got = actual["out"]
    assert max(abs(a - b) for a, b in zip(got, expected)) < 2e-2
    assert any(decision.reason.startswith("batch_norm") for decision in pipeline.normalization)
