"""Rank-3 attention keeps the head axis without a reshape copy."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

SOURCE = """
pipeline p {
  input q:bf16[4, 1, 32]
  input k:bf16[2, 8, 32]
  input v:bf16[2, 8, 32]
  out = attention(q, k, v)
  output out
}
"""


def test_3d_attention_types_and_reference_runs():
    program = compile_source(SOURCE).pipelines[0]
    assert program.symbols.lookup("out").type.shape == ["4", "1", "32"]
    values = reference.run(program.reference)
    assert len(values["out"]) == 4 * 32
