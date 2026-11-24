# Cached decode block

`examples/llama3_decode_block.fgpu` composes the production decode path in one
subgraph:

```text
x -> RMSNorm -> QKV -> RoPE -> cache_write -> attention
  -> output projection -> residual
  -> RMSNorm -> SwiGLU MLP -> residual
```

It returns three values:

- `out`: next hidden state `[1, 64]`;
- `k_out`, `v_out`: updated caches `[1, 8, 64]`.

The caller supplies:

- `loc`: the flat cache slot for the current token;
- `rope_cos/rope_sin`: the table for the current position;
- the previous `k_cache/v_cache`.


## Batched GQA variant

`examples/llama3_decode_block_gqa.fgpu` is the serving-shaped variant:

- batch `B=2`;
- 4 query heads, 2 KV heads (GQA group size 2);
- depth 64;
- 8 cache slots;
- `loc` is `[B]`, written with `cache_write_hsd_batch`;
- outputs `out [2,64]`, `k_out/v_out [2,2,8,64]`.

It compiles to 18 kernels. H100 smoke test: all outputs finite; batch 0 slot
1 and batch 1 slot 5 are written; untouched slots stay zero.

Still outside this example: causal mask (decode reads one slot), paged
allocation inside the compiler, and dynamic batching.

## Scope

This is a single-head, batch-1 decode block with a fixed 8-slot cache. It is
the shape used by the example and tests, not a full LLaMA-3 layer:

- no GQA;
- no causal mask (decode sees only the current slot);
- no batched requests;
- no paged allocator inside the compiler.

`PagedKVCache` can supply `loc` and the block table on the host; batching and
GQA remain separate extensions.

## Verified

H100 smoke test: outputs `(1,64)`, `(1,8,64)`, `(1,8,64)`, all finite; the
selected cache slot is written and untouched slots stay zero.
