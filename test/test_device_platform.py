"""Device-aware GEMM policy is a shared-memory decision, not a name check."""
from flowgpu.compiler.backend.platform_configs import large_n_matmul_policy


def test_large_n_tile_requires_room_for_three_stages():
    assert large_n_matmul_policy(96 * 1024) == {}
    assert large_n_matmul_policy(149 * 1024) == {}
    policy = large_n_matmul_policy(150 * 1024)
    assert policy["matmul_block_n"] == 256
    assert policy["matmul_num_stages"] == 3


def test_h100_class_shared_memory_selects_large_n():
    assert large_n_matmul_policy(227 * 1024)["matmul_block_n"] == 256


def test_device_tile_is_clamped_to_literal_extents():
    from flowgpu.compiler.backend.platform_configs import resolve_platform
    from flowgpu.compiler.compile import compile_source

    config = resolve_platform("cuda", matmul_block_m=128, matmul_block_n=256,
                              matmul_block_k=64, matmul_num_warps=8,
                              matmul_num_stages=3)
    small = """
    pipeline p {
      input a: bf16[64, 128]
      input b: bf16[128, 64]
      out = matmul(a, b)
      output out
    }
    """
    program = compile_source(small, platform_config=config).pipelines[0]
    assert program.triton.kernels[0].tile == (64, 64, 64)

    large = """
    pipeline p {
      input a: bf16[2048, 4096]
      input b: bf16[4096, 14336]
      out = matmul(a, b)
      output out
    }
    """
    program = compile_source(large, platform_config=config).pipelines[0]
    assert program.triton.kernels[0].tile == (128, 256, 64)
