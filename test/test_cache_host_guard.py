"""cache_write_hsd rejects an out-of-range loc before launching."""
from flowgpu.compiler.compile import compile_source


def test_cache_write_hsd_emits_a_loc_range_guard():
    source = """
    pipeline p {
      input cache: bf16[1,8,64]
      input token: bf16[1,64]
      input loc: int32[1]
      out = cache_write_hsd(cache, token, loc)
      output out
    }
    """
    host = compile_source(source).pipelines[0].triton.host
    assert "outside the cache range" in host
