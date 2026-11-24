# End-to-end token loop

`examples/generate_loop.py` closes the loop from token id back to token id:

```text
token id
  -> embedding row
  -> cached decode block (attention + KV cache + MLP)
  -> LM head
  -> greedy sampling
  -> next token id
```

Host pieces used:

- `PagedKVCache` and `BlockTable` for the KV slots;
- the compiled `llama3_decode_block` for attention/MLP/cache;
- the compiled `lm_head` for logits;
- `greedy` from `sampling`.

H100 run: 8 decode steps, cache locations `[0..7]`, tokens
`[3, 105, 43, 17, 95, 10, 86, 60, 7]`, all inside the 128-token vocabulary.

## Scope

This is an integration demo with random, scaled weights: the token sequence
is not meaningful language. It is single-request, depth 64, vocabulary 128 and
uses greedy sampling only. It proves the runtime chain, not model quality.

A real model still needs tokenizer, trained weights, proper LM head, stop
conditions and a continuous-batching policy.
