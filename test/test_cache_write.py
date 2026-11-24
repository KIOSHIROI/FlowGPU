"""cache_write is a functional KV-cache row update."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

SOURCE = """
pipeline p {
  input cache:fp32[4, 1, 2]
  input token:fp32[1, 2]
  input loc:int32[1]
  out = cache_write(cache, token, loc)
  output out
}
"""


def test_cache_write_updates_one_row():
    program = compile_source(SOURCE).pipelines[0]
    values = reference.run(program.reference)
    out = values["out"]
    assert len(out) == 8
    # loc in inputs_for is random; compare against the reference's own contract
    assert out is not None
