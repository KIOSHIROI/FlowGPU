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

## Scope

Single-head, batch 1, fixed 8-token prompt and 32 slots. The causal attention
runs over the new prompt tokens; previously cached tokens are not read back in
this example. GQA, dynamic prompt lengths and paged allocation remain separate
extensions.
