# Prefill + decode + sampling closed loop

`examples/prefill_decode_generate.py` is the full runtime chain:

```text
prompt token ids
  -> embedding rows
  -> cached prefill (slots 0..7)
  -> last hidden state
  -> LM head
  -> greedy sampling
  -> cached decode (slots 8..11)
  -> LM head / sampling for each next token
```

Components used:

- `PagedKVCache` and `BlockTable` for the 32-slot cache;
- `llama3_prefill_block_cached` for causal prefix prefill;
- `llama3_decode_block_masked32` for masked incremental decode;
- `lm_head` for logits;
- `sampling.greedy` for token selection.

H100 run: prompt `[0..7]`, decode locations `[8..11]`, generated tokens
`[88, 109, 50, 75, 59]`, all inside the 128-token vocabulary.

## What v1.0.0 means

The runtime pieces now form a complete token loop: cache, prefill, decode,
logits and sampling. This is **not** a complete serving engine or a trained
model:

- weights are random and scaled; the generated text is not meaningful;
- single request, single head, fixed depths;
- no tokenizer, stop condition, priority or KV eviction;
- the real Llama 3 MLP benchmark remains a separate subgraph measurement.

It is the integration milestone, not a claim of model quality or production
serving completeness.
