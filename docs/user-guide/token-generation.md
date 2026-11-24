# Token generation

`DecodeSession` owns the KV cache and runs the compiled decode-step pipeline:

```python
from flowgpu.compiler.runtime.decode import DecodeSession

session = DecodeSession(source, "quickstart_decode_gqa_step", ".flowgpu-cache")
out = session.step(q, k_new, v_new)
session.reset()
```

The pipeline must expose `k2`, `v2` and `out` (see
`examples/quickstart/decode_gqa_step.fgpu`). Each step writes one KV row,
reshapes the head-major cache and runs native attention.

## Measured loop (H100 GPU1, shared)

32 steps, 4 warmup:

| path | per token | tokens/s |
|---|---:|---:|
| functional cache write | 105.53 µs | 9,475.8 |
| prepared functional | 102.30 µs | 9,774.7 |
| **caller-owned cache, in-place row write** | **72.11 µs** | **13,867.4** |

The in-place path uses `DecodeSession(inplace=True)`: the caller's `[H,S,D]`
cache is updated with one row copy and the compiled attention pipeline only
reads it. This removes the full-cache functional copy. It is still not a
finished throughput claim; the remaining cost is Python/host dispatch and the
absence of a captured multi-step graph.

Raw data: [`token-loop-h100-gpu1-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/token-loop-h100-gpu1-20261124.json), [`token-loop-inplace-h100-gpu1-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/token-loop-inplace-h100-gpu1-20261124.json).
