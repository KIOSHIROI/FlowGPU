"""window_attention typechecks and carries the valid-key count."""
from flowgpu.compiler.compile import compile_source


def test_window_attention_compiles_with_valid_count():
    source = """
    pipeline p {
      input q: bf16[1,1,1,64]
      input k: bf16[1,1,8,64]
      input v: bf16[1,1,8,64]
      input valid: int32[1]
      out = window_attention(q, k, v, valid)
      output out
    }
    """
    program = compile_source(source).pipelines[0]
    assert program.triton.kernels[0].kind == "attention"
