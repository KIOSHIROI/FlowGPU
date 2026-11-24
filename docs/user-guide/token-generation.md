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

| metric | value |
|---|---:|
| per token | **105.53 µs** |
| tokens/s | **9,475.8** |

This is the **unoptimized functional-cache loop**: each step copies the entire
16K cache and pays Python/host dispatch. It is a correctness-capable generation
path, not a throughput claim. The next step is an in-place cache ABI and a
prepared/graph decode loop.

Raw data: [`token-loop-h100-gpu1-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/token-loop-h100-gpu1-20261124.json).
