"""PagedKVCache allocation, write/read and location arithmetic."""
import pytest


def test_allocate_write_read_and_release():
    torch = pytest.importorskip("torch")
    from flowgpu.compiler.runtime.paged_cache import PagedKVCache

    cache = PagedKVCache(blocks=2, page_size=4, heads=1, depth=2, dtype=torch.float32, device="cpu")
    block = cache.allocate()
    assert block == 0
    assert cache.location(block, 2) == 2
    cache.write(block, 2, torch.ones(1, 2), torch.full((1, 2), 3.0))
    k, v = cache.read(block, 2, 1)
    assert torch.equal(k, torch.ones(1, 2))
    assert torch.equal(v, torch.full((1, 2), 3.0))
    cache.release(block)
    assert cache.free_blocks == 2


def test_page_bounds_and_reuse():
    torch = pytest.importorskip("torch")
    from flowgpu.compiler.runtime.paged_cache import PagedKVCache

    cache = PagedKVCache(blocks=1, page_size=2, heads=1, depth=1, dtype=torch.float32, device="cpu")
    block = cache.allocate()
    with pytest.raises(ValueError, match="offset"):
        cache.location(block, 2)
    with pytest.raises(ValueError, match="read"):
        cache.read(block, 1, 2)
    cache.release(block)
    assert cache.allocate() == block
