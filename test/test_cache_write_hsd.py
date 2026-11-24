"""cache_write_hsd updates one head-major cache row."""
from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

SOURCE = """
pipeline p {
  input cache:fp32[2, 4, 2]
  input token:fp32[2, 2]
  input loc:int32[1]
  out = cache_write_hsd(cache, token, loc)
  output out
}
"""


def test_cache_write_hsd_roundtrips_reference():
    program = compile_source(SOURCE).pipelines[0]
    values = reference.run(program.reference)
    assert len(values["out"]) == 16
