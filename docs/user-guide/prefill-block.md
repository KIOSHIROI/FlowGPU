# Cached prefill block

`examples/llama3_prefill_block.fgpu` is the prompt-side counterpart of the
cached decode block. It takes 8 prompt tokens, runs causal attention over the
new tokens, writes the new K/V rows into a contiguous cache range, and applies
the full MLP/residual block.

- `x`: `[8, 64]` (8 tokens, depth 64);
- cache: `[1, 32, 64]` for K and V;
- `loc`: first cache slot for the range write;
- outputs: `out [8,64]`, `k_out/v_out [1,32,64]`.

It compiles to 20 kernels. H100 smoke test: all outputs finite; slots
`loc..loc+7` are written; slots before and after the range stay zero.


## Cached-prefix variant

`examples/llama3_prefill_block_cached.fgpu` attends to the cached prefix plus
the new prompt block. It uses `cached_attention(q, k, v, valid, prefix)`:

- `valid = loc + 8` is the written-key count;
- `prefix = loc` is the cached-token count before the block;
- query i sees keys `j < valid` and `j <= prefix + i`.

H100 check: with `loc=4`, filling slots 12..31 with arbitrary values leaves
`out` bit-identical (max diff 0.0); outputs `(8,64)`, `(1,32,64)`,
`(1,32,64)` are finite. The plain prefill example only attends to the new
tokens; this variant is the one that matches a real cached prefill.


## Prefill to decode on the same cache

`examples/prefill_decode_loop.py` uses one 32-slot `PagedKVCache`:

1. `BlockTable` reserves slots 0..7 for an 8-token prompt;
2. `llama3_prefill_block_cached` writes those slots and returns the last
   hidden state;
3. `llama3_decode_block_masked32` (same 32-slot cache shape) appends four
   decode tokens at locations 8..11;
4. each decode step masks positions past the written count.

H100 run: locations `[0..11]`, decode output `(1,64)` finite, and all twelve
slots are written. The two pipelines share one cache layout; the host owns the
position bookkeeping.

## Scope

Single-head, batch 1, fixed 8-token prompt and 32 slots. The causal attention
runs over the new prompt tokens; previously cached tokens are not read back in
this example. GQA, dynamic prompt lengths and paged allocation remain separate
extensions.
