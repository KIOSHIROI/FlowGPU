# Operator coverage

This is a **static checklist audit**, not a correctness or performance claim.
It answers whether a representative LLM-inference operation can be named in the
FlowGPU frontend and which lowering kind implements it.

Regenerate with:

```sh
python tools/operator_coverage.py
python tools/operator_coverage.py --markdown
python tools/operator_coverage.py --json
```

The current checklist has **44 present / 2 missing** of 46 representative
operations; the frontend registers 91 operators across 18 kernel kinds.

## Missing in the current checklist

- `sort`
- symbolic dynamic-shape dispatch (per-shape specialization exists)

These are **gaps, not regressions**: the audit records what a model trace would
need next. The remaining items are larger design gaps; convolution,
batch norm, sort and symbolic dispatch are larger design items.

## Checklist

| category | operation | FlowGPU | status | kind |
|---|---|---|---|---|
| pointwise | add | `add` | present | - |
| pointwise | mul | `mul` | present | - |
| pointwise | sub | `sub` | present | - |
| pointwise | div | `div` | present | primitive |
| pointwise | exp | `exp` | present | primitive |
| pointwise | log | `log` | present | primitive |
| pointwise | sqrt | `sqrt` | present | primitive |
| pointwise | rsqrt | `rsqrt` | present | primitive |
| pointwise | sigmoid | `sigmoid` | present | primitive |
| pointwise | silu | `silu` | present | - |
| pointwise | gelu | `gelu` | present | primitive |
| pointwise | tanh | `tanh` | present | primitive |
| pointwise | abs | `abs` | present | primitive |
| pointwise | neg | `neg` | present | primitive |
| pointwise | pow | `pow` | present | primitive |
| pointwise | where | `where` | present | primitive |
| reduction | sum | `sum_last` | present | primitive |
| reduction | max | `max_last` | present | primitive |
| reduction | mean | `mean_last` | present | - |
| reduction | argmax | `argmax_last` | present | reduce_with_index |
| reduction | softmax | `softmax` | present | rowwise |
| norm | rmsnorm | `rms_norm` | present | rowwise |
| norm | layernorm | `layer_norm` | present | - |
| norm | batch_norm | `batch_norm` | present | - |
| linear algebra | matmul | `matmul` | present | matmul |
| linear algebra | conv2d | `conv2d` | present | - |
| linear algebra | embedding lookup | `gather_rows` | present | primitive |
| attention | scaled dot-product attention | `attention` | present | attention |
| attention | causal attention | `causal_attention` | present | attention |
| position | rope | `rope` | present | rope |
| layout | reshape | `reshape` | present | layout |
| layout | transpose | `transpose` | present | transpose |
| layout | concat | `concat` | present | layout |
| layout | split | `split` | present | layout |
| layout | pad+concat | `pad_cat` | present | pad_cat |
| gather | row gather | `gather_rows` | present | primitive |
| gather | last-axis gather | `gather_last` | present | axis_gather |
| gather | scatter by index | `scatter_by_index` | present | scatter_by_index |
| sort/search | top-k | `topk_last` | present | topk |
| sort/search | sort | `sort` | missing | - |
| scan | cumsum | `cumsum` | present | scan |
| scan | segmented cumsum | `segmented_cumsum` | present | - |
| state | cache store | `store_cache_4d` | present | cache_store |
| state | linear decode | `linear_decode_forward` | present | linear_decode |
| state | recurrence | `gated_delta_recurrence` | present | recurrence |
| shape | dynamic symbolic shapes | `dynamic_shape` | missing | - |
