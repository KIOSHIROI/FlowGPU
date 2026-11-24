"""cache_write_hsd_batch writes one row per (batch, head)."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

SOURCE = """
pipeline p {
  input cache:fp32[2, 2, 4, 2]
  input token:fp32[2, 2, 2]
  input loc:int32[2]
  out = cache_write_hsd_batch(cache, token, loc)
  output out
}
"""


def test_cache_write_batch_roundtrips_reference():
    program = compile_source(SOURCE).pipelines[0]
    values = reference.run(program.reference)
    assert len(values["out"]) == 32
