"""sort_last is a topk(k=width) composition over literal rows."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.operators import OPERATORS


def test_sort_last_is_composed_and_roundtrip_sorts_ascending():
    assert "sort_last" in OPERATORS
    source = """
    pipeline p {
      input x:fp16[4, 8]
      out = sort_last(x)
      output out
    }
    """
    pipeline = compile_source(source).pipelines[0]
    values = reference.run(pipeline.reference)["out"]
    inputs = reference.inputs_for(pipeline.reference, {})[0][3]
    expected = []
    for row in range(4):
        expected.extend(sorted(inputs[row * 8:(row + 1) * 8]))
    assert max(abs(a - b) for a, b in zip(values, expected)) < 2e-2
    assert any(decision.reason.startswith("sort_last") for decision in pipeline.normalization)
