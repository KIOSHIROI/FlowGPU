# Paged KV cache

`PagedKVCache` is the host-side block allocator for a serving loop. FlowGPU
still sees a flat K/V tensor and a `loc` value; the cache owns block reuse and
the block table.

```python
from flowgpu.compiler.runtime.paged_cache import PagedKVCache

cache = PagedKVCache(blocks=64, page_size=16, heads=8, depth=128)
block = cache.allocate()
location = cache.location(block, 0)     # flat row for store_cache_4d / decode
cache.write(block, 0, k_row, v_row)
k, v = cache.read(block, 0, 1)
cache.release(block)
```

- allocation unit is a block; a token is `block * page_size + offset`;
- `allocate()` zeroes the reused block;
- `release()` returns it to the free list;
- `write/read` validate `[heads, depth]` and page bounds;
- no scheduler, request lifecycle or eviction policy is implied.


## Request block table

`BlockTable` maps request ids to blocks and appends one token at a time:

```python
from flowgpu.compiler.runtime.block_table import BlockTable

table = BlockTable(cache)
table.add("req-1")
location = table.append("req-1")   # allocates a page on the first token
table.locations("req-1")           # flat locations for the request so far
table.release("req-1")             # returns all blocks to the cache
```

It does not schedule requests or decode; the serving loop decides which
request appends next and passes `location` to the cache-aware pipeline.

A cached decode block example is in [Cached decode block](decode-block.md).

The allocator is deliberately separate from the compiler: a serving engine can
own request scheduling and block tables while FlowGPU compiles the cache-aware
attention and MLP subgraphs.
