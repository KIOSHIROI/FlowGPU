# Why the MLP is slower than eager, and what to do

The example is `examples/mlp.fgpu`, `B=64, D=256, H=512`:

```text
h  = matmul(x, w1)   # 64 x 256 @ 256 x 512
h2 = silu(h)
y  = matmul(h2, w2)  # 64 x 512 @ 512 x 256
```

There are two separate causes, and they need different fixes.

## 1. The default GEMM tile is wrong for M=64

The CUDA policy measures a 128x128x64 tile on large GEMMs. With M=64 that tile
spends half its rows on padding and launches only four programs for
`64x512` (and two more for the second GEMM) on a 132-SM H100.

Profiler measurement of the two kernels, before and after the small-M policy:

| kernel | 128x128x64 default | 16x64x64 small-M |
|---|---:|---:|
| first GEMM + silu | 5.87 µs | **2.42 µs** |
| second GEMM | 5.94 µs | **3.03 µs** |
| GPU total | 11.81 µs | **5.45 µs** |

The small-M tile is selected only for CUDA, literal `M <= 64`, `K >= 64`,
`N >= 128`, and when the platform still uses the default 128-row floor. Large
GEMMs keep the measured 128x128x64 policy.

## 2. The default host call pays guards and allocation per call

The generated host validates shapes, dtypes, contiguity and index bounds and
allocates two output buffers on every call. With a per-call
`torch.cuda.synchronize()` (the way a "complete call" is usually measured):

| path | per-call median | loop, no per-call sync |
|---|---:|---:|
| FlowGPU default, old tile | 63.9 µs | 33.4 µs |
| **FlowGPU default, small-M tile** | **39.1 µs** | **33.2 µs** |
| FlowGPU prepared plan | 32.9 µs | — |
| **FlowGPU CUDA Graph replay** | **11.9 µs** | — |
| eager bf16 | 31.3 µs | 23.0 µs |
| torch.compile | 59.5 µs | 47.1 µs |

The same case in the M2 device harness (CUDA events, no per-call sync) reads:

| flowgpu | prepared | graph | eager | inductor |
|---:|---:|---:|---:|---:|
| 24.02 | 18.04 | **6.77** | 34.91 | 54.15 |

So the reported 47/45 µs against 23-27 µs is real for the guarded per-call
path, but the graph path on the same build is **11.9 µs**, faster than eager
(31.3 µs). The default host is the price of the safety checks, not of the
generated GEMM.

## What to use

```python
from flowgpu.compiler.runtime.graph import prepare_graph_pipeline

runner = prepare_graph_pipeline(
    source, pipeline="mlp", args=(x, w1, w2),
    cache_dir=".flowgpu-cache", shape_bindings={"B": 64, "D": 256, "H": 512})
runner(x, w1, w2)   # CUDA Graph replay path
```

For latency-sensitive serving, use the graph path; keep the guarded call for
validation and testing. The small-M tile fix applies to both.
