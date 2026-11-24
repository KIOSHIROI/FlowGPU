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

32 steps, 4 warmup, tokens precomputed **outside** the timed loop (a real model
produces them on device; allocating `torch.randn` inside the loop measures the
allocator, not decode):

| path | per token | tokens/s |
|---|---:|---:|
| compiled in-place stride | **38.76 µs** | **25,803.0** |
| **in-place + CUDA Graph replay** | **20.83 µs** | **48,002.3** |

The graph path is 1.86x the compiled in-place loop. It captures the
attention-only pipeline with caller-owned cache and query buffers:

```python
session = DecodeSession(
    source, "quickstart_decode_gqa_attention_nocopy",
    ".flowgpu-cache", inplace=True, graph=True)
out = session.step(q, k_new, v_new)
```

The earlier 72.11 µs/token figure is retained only as a historical number: it
allocated random tensors inside the timed loop and is not comparable. The graph
session captures on the first step, so warm it up before measuring and do not
reset between warmup and timing.

Raw data: [`token-loop-graph-h100-gpu1-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/token-loop-graph-h100-gpu1-20261124.json).
