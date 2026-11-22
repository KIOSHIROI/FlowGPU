"""`neg` is a value-layer unary operator with a reference and Triton lowering."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.operators import OPERATORS

SOURCE = """
pipeline p {
  input x:fp16[8]
  out = neg(x)
  output out
}
"""


def test_neg_is_registered_and_reference_matches():
    assert "neg" in OPERATORS
    compiled = compile_source(SOURCE)
    assert compiled.pipelines[0].pipeline.name == "p"
    values = reference.run(compiled.pipelines[0].reference)
    expected = [-value for value in reference.inputs_for(compiled.pipelines[0].reference, {})[0][3]]
    assert values["out"] == expected
