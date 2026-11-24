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


The prefill counterpart is in [Cached prefill block](prefill-block.md).

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



## Masked variant (use this for incremental decode)

`examples/llama3_decode_block_masked.fgpu` replaces the attention operator
with explicit 2D matmuls plus a host-supplied `mask [1,8]` and
`q_scale [1,64]`:

- `mask` is 0 for valid slots and `-1e9` for unused slots;
- `q_scale` is `1/sqrt(depth)` (0.125 for depth 64);
- scores are `q @ K^T`, masked, softmaxed, then multiplied by V.

H100 check: with slots 0..3 valid, filling slots 4..7 with arbitrary values
leaves `out` **bit-identical** (max diff 0.0). The unmasked example is only
valid when every cache slot holds a real token; incremental decode must use
the masked variant.

## Host loop wiring

`examples/decode_loop.py` drives 8 decode steps for one request using the masked variant:

1. `BlockTable.add("req-1")`;
2. each step calls `table.append("req-1")` to get the next flat cache slot;
3. `PagedKVCache` storage is reshaped to the block's `[1,S,D]` cache input;
4. the block returns `out`, `k_out`, `v_out`;
5. the updated K/V are copied back into the paged storage and `out` becomes
   the next `x`.

H100 run: locations `[0..7]`, output `(1,64)` finite, all 8 cache slots
written. This is the minimal end-to-end host chain; sampling and request
eviction are still separate.


## Dynamic batching loop

`examples/decode_loop_batched.py` combines all three host pieces over the GQA
block (batch 2, 4 query heads, 2 KV heads, 8 slots):

- `DynamicBatch(capacity=2)` packs active requests and pads empty rows;
- each request owns a `PagedKVCache` and a `BlockTable`;
- each step appends one cache location per active request, stacks the per-row
  caches into the block's `[2,2,8,64]` input, runs, and copies the updated
  rows back.

H100 run: after 3 steps both requests are at locations `[0,1,2]`; after
removing `r0` and adding `r2`, the next step gives `r1=[0,1,2,3]`,
`r2=[0]`, and all outputs are finite.

This is a scheduling demo, not a continuous-batching policy: there is no
priority, preemption or KV eviction.

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
