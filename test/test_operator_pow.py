"""`pow` is a value-layer binary operator with reference and Triton lowering."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.core.operators import OPERATORS

SOURCE = """
pipeline p {
  input x:fp16[8]
  input e:fp16[8]
  out = pow(x, e)
  output out
}
"""


def test_pow_is_registered_and_reference_is_consistent():
    assert "pow" in OPERATORS
    compiled = compile_source(SOURCE)
    program = compiled.pipelines[0].reference
    values = reference.run(program)
    inputs = reference.inputs_for(program, {})
    import math
    expected = []
    for a, b in zip(inputs[0][3], inputs[1][3]):
        try:
            expected.append(math.pow(a, b))
        except ValueError:
            expected.append(math.nan)
    got = values["out"]
    for actual, wanted in zip(got, expected):
        assert (math.isnan(actual) and math.isnan(wanted)) or abs(actual - wanted) < 2e-3
