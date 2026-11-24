# Continuous batching

`ContinuousBatchSession` keeps a slot table over the compiled GPU batch and
admits/retires requests dynamically:

```python
from flowgpu.compiler.runtime.batching import ContinuousBatchSession, RequestState

session = ContinuousBatchSession(
    source, "quickstart_decode_batch_attention",
    ".flowgpu-cache", capacity=4, inplace=True)
session.admit(RequestState(request_id=7, max_tokens=8))
out = session.step({7: (q, k_new, v_new)})
```

- GPU batch is the compiled literal capacity (4);
- free slots are zeroed on admission;
- a finished request frees its slot and the next request enters on the next
  iteration;
- the in-place path writes each request's row into its own `[B,H,S,D]` cache.


## Prefill-aware admission

A request can carry a prompt; on admission its KV rows are written into the
freed slot before decoding starts:

```python
session.admit(RequestState(
    request_id=7, max_tokens=24, prompt_length=16,
    prompt_k=prompt_k, prompt_v=prompt_v))
```

### Measured (H100 GPU1, shared)

16 requests, 16-token prompt, 8 decode tokens, capacity 4:

| metric | value |
|---|---:|
| prompt tokens | 256 |
| decode tokens | 128 |
| total schedule | 5.537 ms |
| per decode token | **43.26 µs** |
| decode throughput | **23,116.7 tokens/s** |

The prompt copy is a runtime row copy, not the compiled prefill pipeline, so the
merged schedule is slower per decode token than the 30,998 tok/s decode-only
schedule. It is the admission/prefill scheduling primitive; compiled prefill and
graph capture are the next steps.

## Measured schedule (H100 GPU1, shared)

16 requests × 8 tokens, capacity 4:

| path | per token | tokens/s |
|---|---:|---:|
| single in-place session | 72.11 µs | 13,867.4 |
| continuous batch, functional cache | 263.05 µs | 3,801.5 |
| **continuous batch, in-place cache** | **32.26 µs** | **30,997.9** |

In-place continuous batching is 2.2x the single-request loop on this schedule.

This is still a padded, fixed-capacity scheduler: no prefill, no KV-aware
admission policy, and no vLLM/TensorRT-LLM comparison. It is the scheduling
primitive, not a serving benchmark claim.

Raw data: [`continuous-batch-h100-gpu1-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/continuous-batch-h100-gpu1-20261124.json).
