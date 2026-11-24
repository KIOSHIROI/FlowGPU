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
