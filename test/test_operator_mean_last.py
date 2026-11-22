"""mean_last is a composed operator: sum_last(x) / sum_last(ones_like(x))."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.operators import OPERATORS

SOURCE = """
pipeline p {
  input x:fp16[2, 4]
  out = mean_last(x)
  output out
}
"""


def test_mean_last_is_composed_and_reference_is_correct():
    assert "mean_last" in OPERATORS
    compiled = compile_source(SOURCE)
    pipeline = compiled.pipelines[0]
    values = reference.run(pipeline.reference)
    inputs = reference.inputs_for(pipeline.reference, {})
    width = len(inputs[0][3]) // 2
    raw = inputs[0][3]
    expected = [sum(raw[row * width:(row + 1) * width]) / width for row in range(2)]
    got = values["out"]
    assert len(got) == 2
    assert max(abs(a - b) for a, b in zip(got, expected)) < 1e-5
    assert any(decision.reason.startswith("mean_last ->") for decision in pipeline.normalization)
