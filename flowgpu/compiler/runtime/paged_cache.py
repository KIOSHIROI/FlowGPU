"""A small host-side paged KV-cache allocator.

The compiler sees a flat cache and a ``loc`` tensor; this class owns block
allocation and the block table for a serving loop. It does not schedule
requests and does not launch kernels; the caller passes ``location(block,
offset)`` to the cache-aware pipeline.
"""
from __future__ import annotations


class PagedKVCache:
    """Fixed-capacity paged cache with per-block allocation.

    Storage is ``[blocks * page_size, heads, depth]`` for both K and V. A block
    is the unit of allocation and reuse; a token lives at
    ``block * page_size + offset``.
    """

    def __init__(self, *, blocks, page_size, heads, depth, dtype=None, device=None):
        import torch
        if blocks <= 0 or page_size <= 0 or heads <= 0 or depth <= 0:
            raise ValueError("blocks, page_size, heads and depth must be positive")
        self.blocks = int(blocks)
        self.page_size = int(page_size)
        self.heads = int(heads)
        self.depth = int(depth)
        self.dtype = dtype or torch.bfloat16
        self.device = device or "cuda"
        shape = (self.blocks * self.page_size, self.heads, self.depth)
        self.k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.v = torch.zeros_like(self.k)
        self._free = list(range(self.blocks))

    @property
    def free_blocks(self) -> int:
        return len(self._free)

    def allocate(self) -> int:
        if not self._free:
            raise MemoryError("paged KV cache has no free blocks")
        block = self._free.pop(0)
        self.k[block * self.page_size:(block + 1) * self.page_size].zero_()
        self.v[block * self.page_size:(block + 1) * self.page_size].zero_()
        return block

    def release(self, block: int) -> None:
        self._check_block(block)
        if block in self._free:
            raise ValueError("block is already free")
        self._free.append(block)
        self._free.sort()

    def location(self, block: int, offset: int) -> int:
        self._check_block(block)
        if not 0 <= offset < self.page_size:
            raise ValueError("offset is outside the page")
        return block * self.page_size + offset

    def write(self, block: int, offset: int, k_row, v_row) -> None:
        import torch
        location = self.location(block, offset)
        if tuple(k_row.shape) != (self.heads, self.depth) or tuple(v_row.shape) != (self.heads, self.depth):
            raise ValueError("K/V row shape must be [heads, depth]")
        self.k[location].copy_(k_row.to(dtype=self.dtype, device=self.device))
        self.v[location].copy_(v_row.to(dtype=self.dtype, device=self.device))

    def read(self, block: int, offset: int, count: int):
        self._check_block(block)
        if count <= 0 or offset < 0 or offset + count > self.page_size:
            raise ValueError("read must stay inside one page")
        start = block * self.page_size + offset
        return self.k[start:start + count], self.v[start:start + count]

    def _check_block(self, block: int) -> None:
        if not isinstance(block, int) or not 0 <= block < self.blocks:
            raise ValueError("unknown block")


__all__ = ["PagedKVCache"]
