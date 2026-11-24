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
