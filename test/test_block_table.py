"""BlockTable maps requests to pages and flat locations."""


class _FakeCache:
    page_size = 2

    def __init__(self):
        self.next = 0
        self.free = []

    def allocate(self):
        if self.free:
            return self.free.pop(0)
        block = self.next
        self.next += 1
        return block

    def release(self, block):
        self.free.append(block)

    def location(self, block, offset):
        return block * self.page_size + offset


def test_append_allocates_pages_and_locations():
    from flowgpu.compiler.runtime.block_table import BlockTable

    table = BlockTable(_FakeCache())
    table.add("a")
    assert [table.append("a") for _ in range(5)] == [0, 1, 2, 3, 4]
    assert table.length("a") == 5
    assert table.locations("a") == [0, 1, 2, 3, 4]


def test_release_returns_blocks_and_unknown_is_rejected():
    from flowgpu.compiler.runtime.block_table import BlockTable

    cache = _FakeCache()
    table = BlockTable(cache)
    table.add("a")
    table.append("a")
    table.append("a")
    table.append("a")
    table.release("a")
    assert "a" not in table
    assert cache.free == [0, 1]
