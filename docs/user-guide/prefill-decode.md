# Prefill + decode

The prompt path writes a contiguous KV run and computes causal GQA attention in
one compiled pipeline; the decode path then continues from that cache.

```python
result = prefill(q, k_new, v_new, k_cache, v_cache, loc)  # k2, v2, attn
session.seed(result["k2"], result["v2"], prompt_length)
session.step(q_token, k_token, v_token)
```

## Measured (H100 GPU1, shared)

Prompt 16 tokens, decode 32 tokens, warmed:

| stage | value |
|---|---:|
| prefill | **0.564 ms** for 16 tokens |
| decode | **83.71 µs/token** |
| decode throughput | **11,945.9 tokens/s** |

Prefill and decode telemetry are separate: prefill is a single call over the
prompt, decode is the single-request in-place loop. No prefill graph capture, no
batching and no vLLM/TensorRT-LLM comparison are claimed here.

Raw data: [`prefill-decode-h100-gpu1-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/prefill-decode-h100-gpu1-20261124.json).
