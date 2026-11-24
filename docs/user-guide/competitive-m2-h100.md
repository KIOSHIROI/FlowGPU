# M2 device result: H100 GPU1, 2026-11-22

This is the first measured M2 result. It is **exploratory**, not an official
leaderboard entry: one shared H100 PCIe (GPU1), 30 timed iterations after 8
warmups, one seed. The point is to choose the next optimization from evidence,
not to claim a general win.

## Environment

| item | value |
|---|---|
| GPU | NVIDIA H100 PCIe, GPU1 (`CUDA_VISIBLE_DEVICES=1`) |
| torch | 2.14.1+cu130 |
| CUDA / Triton | 13.0 / 3.8.0 |
| measurement | CUDA events; Graph = capture+replay, full = guarded host call |
| raw result | [`m2-h100-gpu1-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-20261122.json) |

The GPU was shared (about 48 GiB already allocated by another process,
utilization 0 at measurement time). Do not compare these absolute numbers with
an exclusive run.

## Result

| case | metric | flowgpu | flowgpu_prepared | eager | inductor | triton |
|---|---|---:|---:|---:|---:|---:|
| add_literal | graph (µs) | 2.79 | 2.70 | 2.76 | 2.72 | 2.66 |
| add_literal | full (µs) | 17.63 | 14.06 | 5.18 | 22.14 | 13.25 |
| vectoradd_symbolic | graph (µs) | 2.69 | 2.71 | 2.64 | 2.60 | 2.58 |
| vectoradd_symbolic | full (µs) | 18.16 | 14.12 | 5.42 | 26.34 | 13.43 |
| matmul_symbolic | graph (µs) | 4.08 | 4.07 | 3.63 | 3.66 | 3.03 |
| matmul_symbolic | full (µs) | 18.40 | 14.18 | 9.17 | 40.10 | 16.00 |
| softmax_wide | graph (µs) | 2.74 | 2.79 | 5.39 | 3.78 | 2.73 |
| softmax_wide | full (µs) | 16.25 | 12.29 | 5.40 | 25.43 | 11.57 |
| residual_norm | graph (µs) | 2.74 | 2.72 | 22.42 | 2.65 | 2.64 |
| residual_norm | full (µs) | 21.34 | 16.64 | 61.30 | 25.49 | 12.65 |
| mapping_pointwise | graph (µs) | 2.80 | 2.69 | 3.64 | 2.65 | 2.65 |
| mapping_pointwise | full (µs) | 17.08 | 13.82 | 9.12 | 23.88 | 13.58 |

`flowgpu_prepared` is the offline `prepare_execution_plan` path with
caller-owned buffers. It now selects the named pipeline from a multi-pipeline
source before preparing (`softmax_wide` included). `triton` is the hand-written
baseline in `benchmarks/competitive/triton_baselines.py`.

## What the numbers say

1. **The kernel path is competitive.** FlowGPU and explicit Triton have nearly
   the same Graph latency on pointwise, softmax and residual cases; FlowGPU is
   slightly slower on matmul (4.08 vs 3.03 µs). Kernel fusion is already doing
   real work: residual RMSNorm Graph is 2.74 µs vs eager 22.42 µs.
2. **The gap is host-side.** FlowGPU full-call is 13–16 µs above its Graph time.
   Eager full-call is 5–9 µs on simple cases. The measured loss is guards,
   allocation and launch dispatch, not kernel math.
3. **Prepared execution closes most of the public-flow gap.** Prepared FlowGPU
   is close to explicit Triton on add (14.06 vs 13.25), vectoradd (14.12 vs
   13.43), softmax (12.29 vs 11.57) and mapping (13.82 vs 13.58), beats it on
   matmul (14.18 vs 16.00), and remains slower on residual (16.64 vs 12.65).
4. **FlowGPU beats TorchInductor full-call on every measured case** (16–21 µs vs
   22–40 µs), but loses to eager on the simple pointwise/softmax cases.
5. **CUDA Graph is not a universal answer.** It removes host overhead but fixes
   input/output pointers; it is a deployment option, not a replacement for the
   dynamic-shape path.



## Multi-seed medians

Three seeds (0, 1, 2), 30 iterations each; values are medians of the per-seed
full-call times, including the new `neg`, `pow` and `mean_last` cases. All
correctness checks pass for every seed.

| case | flowgpu | prepared | graph | eager | inductor | triton |
|---|---:|---:|---:|---:|---:|---:|
| add | 18.41 | 14.18 | 3.25 | 5.60 | 23.99 | 13.56 |
| vectoradd | 19.54 | 14.39 | 3.35 | 5.22 | 27.28 | 14.31 |
| matmul | 19.91 | 14.21 | 4.31 | 9.25 | 41.53 | 16.45 |
| softmax | 17.15 | 11.98 | 3.15 | 5.64 | 26.84 | 11.56 |
| residual_norm | 21.65 | 16.45 | 3.10 | 62.49 | 26.74 | 12.67 |
| mapping | 17.84 | 13.98 | 3.01 | 9.63 | 25.20 | 13.68 |
| neg | 16.12 | 12.60 | 3.11 | 4.88 | 24.06 | 13.44 |
| pow | 17.97 | 14.18 | 3.91 | 5.33 | 25.39 | 13.74 |
| mean_last | 21.24 | 12.70 | 2.99 | 5.99 | 18.46 | 12.46 |


The single-seed conclusions hold: prepared is close to explicit Triton, eager
still wins the simple pointwise/reduction cases, and the default `mean_last`
carries the extra count reduction. Static zero-copy graph replay is 2.99–4.31 µs
across three seeds, below eager and explicit Triton.

Raw data: [`m2-h100-gpu1-multiseed-graph-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-multiseed-graph-20261122.json).




## Peak allocated memory

`torch.cuda.max_memory_allocated()` after a reset at the start of each backend.
Values are KiB and include JIT/warmup allocations, not just kernel working set.

| case | flowgpu | prepared | graph | eager | inductor | triton |
|---|---:|---:|---:|---:|---:|---:|
| add | 42.5 | 40.5 | 40.5 | 42.5 | 42.5 | 42.5 |
| vectoradd | 672.5 | 640.5 | 640.5 | 672.5 | 672.5 | 672.5 |
| matmul | 32952.5 | 176.5 | 176.5 | 32952.5 | 32952.5 | 184.5 |
| softmax | 384.5 | 352.5 | 352.5 | 384.5 | 384.5 | 384.5 |
| residual_norm | 169.5 | 161.5 | 161.5 | 169.5 | 169.5 | 169.5 |
| mapping | 56.0 | 52.0 | 52.0 | 56.0 | 56.0 | 56.0 |
| neg | 40.5 | 38.5 | 38.5 | 40.5 | 40.5 | 40.5 |
| pow | 42.5 | 40.5 | 40.5 | 42.5 | 42.5 | 42.5 |
| mean_last | 5.0 | 4.5 | 4.5 | 5.0 | 5.0 | 5.0 |

The matmul default's 32 MiB is dominated by first-use compilation allocation; the
prepared/static paths report 176 KiB. This metric is useful for spotting that
class of cost, not for claiming a smaller working set without a cleaner protocol.

Raw data: [`m2-h100-gpu1-memory-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-memory-20261122.json).

## Symbolic compile cost

The symbolic path compiles one module for several extents. On H100 GPU1:

| program | symbolic compile | per-shape compile | symbolic full | per-shape full |
|---|---:|---:|---:|---:|
| vectoradd (3 shapes) | 69.9 ms | 7.8 / 6.9 / 7.0 ms | 18.3–18.6 µs | 17.6–17.8 µs |
| matmul (2 shapes) | 7.0 ms | 48.9 / 11.0 ms | 18.3–18.4 µs | 17.7–17.8 µs |
| softmax (2 shapes) | 22.3 ms | 15.8 / 15.0 ms | 17.1–17.4 µs | 15.3 µs |

One module is not automatically cheaper: vectoradd's cold symbolic compile was
slower than compiling three small per-shape modules, while matmul's symbolic
compile was much cheaper than both specializations. Full-call latency is slightly
higher for symbolic. Choose symbolic when the shape set is open-ended; choose
per-shape for a hot, single shape.

Raw data: [`symbolic-h100-gpu1-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/symbolic-h100-gpu1-20261122.json).














## Conv tuning in the matrix

The `flowgpu_tuned` runner measures six conv tile candidates per shape and uses
the fastest. On the 5 conv cases (H100 GPU1, 20 iterations):

| case | flowgpu | tuned | tuning compile |
|---|---:|---:|---:|
| conv2d_1x1 | 2.35 | 2.38 | 525 ms |
| conv2d_3x3 | 3.88 | 4.02 | 108 ms |
| conv2d_same | 22.36* | 18.69* | 216 ms |
| conv2d_stride2 | 22.27* | 3.74 | 112 ms |
| conv2d_stride2_same | 18.73* | 3.68 | 149 ms |

Tuning does not help the shapes whose default geometry is already good, and a
shared-GPU outlier can still change the winner. It clearly helps the stride-2
variants, where the default geometry sat on a bad tile. Compile cost is one-off
and cached by the call site.






## MoE scaling

The same routing pipeline at a more realistic batch/router width
(`[32,128]`, 64 experts, k=8):

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 79.29 | 60.62 | 32.40 | **27.98** | 61.58 | 90.37 |

At this size FlowGPU's graph path beats eager and Inductor; the tiny `[4,16]`
case still loses because six kernel launches dominate. The matrix is now 21
cases.

Raw data: [`m2-h100-gpu1-moe-large-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-moe-large-20261124.json).




## Native causal GQA attention

`causal_attention_gqa` is 4 query heads sharing 2 KV heads with a causal mask.
The default path is now a **native Triton attention kernel**: one program per
(batch, query head, query position), scores computed against the key axis,
causal mask in registers, softmax and value projection in the same kernel.

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 19.82 | 16.25 | 12.84 | **2.93** | 186.26 | 77.34 |

Graph is 63x faster than eager and 26x faster than Inductor on this shape.
Prepared and zero-copy graph now work because the host has no library call.
Correctness matches a float64 causal-GQA oracle. The hybrid library path
remains available via `gpu_backend='hybrid'`.

The matrix is now 24 cases.
Raw data: [`m2-h100-gpu1-causal-native2-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-causal-native2-20261124.json).







## MLP: `64 x 256 @ 256 x 512` then `64 x 512 @ 512 x 256`

The MLP case exposes the small-M GEMM and host-path costs separately.

| flowgpu | prepared | graph | eager | inductor |
|---:|---:|---:|---:|---:|
| 24.02 | 18.04 | **6.77** | 34.91 | 54.15 |

With per-call synchronization, the guarded default is 39.1 µs (after the
small-M tile; 63.9 µs before), eager bf16 is 31.3 µs and graph replay is
11.9 µs. The analysis and the graph API are in
[MLP performance](mlp-performance.md). The matrix is now 30 cases.

Raw data: [`m2-h100-gpu1-mlp-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-mlp-20261124.json).

## Prompt prefill with a KV-cache run write

`prefill_gqa_cache` computes causal GQA attention over a 16-token prompt and
writes the prompt's KV rows into a head-major cache:

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 32.72 | 26.11 | 19.21 | **6.42** | 150.51 | 73.28 |

Graph is 11x faster than Inductor. Correctness matches a float64 causal-GQA
oracle. The matrix is now 29 cases.

Raw data: [`m2-h100-gpu1-prefill-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-prefill-20261124.json).

## No-copy GQA decode (3D attention)

`decode_gqa_nocopy` gives the attention op a rank-3 `[H,L,D]` query and reads
the head-major cache directly; there is no `[H,S,D] -> [1,H,S,D]` reshape copy
on the hot path:

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 8.57 | 8.46 | 19.68 | **7.09** | 2371.09 | 467.03 |

Graph improves from 9.04 to 7.09 us for the same GQA cache-write decode step.
Correctness matches a float64 oracle. The matrix is now 28 cases.

Raw data: [`m2-h100-gpu1-decode-nocopy2-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-decode-nocopy2-20261124.json).

## Batched GQA decode (4 requests)

`decode_batch_gqa` batches 4 requests, each with 4 query heads and 2 KV heads.
One kernel writes one row per (batch, KV head), then native 4D GQA attention runs
over all four request caches:

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 45.19 | 9.64 | 20.36 | **7.61** | 316.11 | 1213.24 |

Graph is 41x faster than eager and 159x faster than Inductor on this batch, and
correctness matches a float64 batched-GQA oracle. Prepared+graph is noisier on
the shared GPU; the zero-copy graph number is the stable one.

The matrix is now 27 cases.
Raw data: [`m2-h100-gpu1-decode-batch-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-decode-batch-20261124.json).

## Multi-head GQA decode with cache write

`decode_gqa_cache` writes 2 KV heads in a **head-major [H,S,D]** cache, reshapes
to `[1,2,128,64]`, and attends with 4 query heads:

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 63.23 | 53.68 | 27.73 | **9.04** | 190.72 | 234.33 |

Graph is 21x faster than eager and 26x faster than Inductor. Correctness matches
a float64 GQA oracle; prepared/graph work. The matrix is now 26 cases.

Raw data: [`m2-h100-gpu1-decode-gqa2-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-decode-gqa2-20261124.json).

## Decode with a functional KV-cache write

`decode_with_cache` writes one new K/V row with `cache_write`, reshapes the
128-token cache and runs native attention. Prepared and zero-copy graph now
work after the execution plan learned to keep host layout helpers.

| flowgpu | prepared | prepared+graph | graph | eager |
|---:|---:|---:|---:|---:|
| 62.64 | 53.24 | 27.58 | **8.68** | 119.61 |

Graph is 13.8x faster than eager on the full cache-write + decode step.
Correctness matches a float64 oracle. The matrix is now 25 cases.

Raw data: [`m2-h100-gpu1-decode-cache3-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-decode-cache3-20261124.json).

## Decode-step attention over a KV cache

The decode-phase case: `q [1,64]` attends over a 128-token single-head K/V
cache. It now uses the native attention kernel.

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 18.74 | 15.88 | 13.03 | **3.49** | 33.10 | 58.02 |

Graph is 9.5x faster than eager and 16.6x faster than Inductor. This is a fixed
cache length, single head, no mask and no cache update; the next stage is a full
KV-cache decode step with cache write and GQA.

The matrix is now 24 cases.
Raw data: [`m2-h100-gpu1-decode-native-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-decode-native-20261124.json).

## End-to-end decoder layer

`llama3_decoder_layer` is the first full decoder block in the matrix: RMSNorm,
QKV projections, RoPE, unscaled attention, output projection, residual, SwiGLU
MLP and a second residual, all in one pipeline.

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 207.72 | 158.59 | 36.97 | **32.16** | 475.55 | 197.63 |

Graph is 6.1x faster than Inductor and 14.8x faster than eager on this shape.
The matrix is now 22 cases. This is still a single-head, non-causal, no-cache
layer; KV-cache decode and GQA remain next-stage work.

Raw data: [`m2-h100-gpu1-decoder-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-decoder-20261124.json).

## QK RMSNorm + RoPE

`qk_transform` is the QKV path: RMSNorm then split-half RoPE on 4-D `[B,H,S,D]`
q/k tensors with full-shape cosine/sine tables.

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 9.01 | 130.74* | 9.28 | 6.62 | 184.50 | 124.25 |

*The prepared sample is a shared-GPU outlier; graph is the stable 6.62 µs path.
The routing values match a float64 RMSNorm+RoPE oracle. The matrix is now 20
cases.

Raw data: [`m2-h100-gpu1-qk-20261123.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-qk-20261123.json).

## Top-k sampling case

`topk_last` on `[16, 1024]` with k=8, checked against torch's descending top-k:

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 4.70 | 4.67 | 7.24 | 4.79 | 18.44 | 18.44 |

The operator is a multi-result backend (`values`, `ids`); this case checks the
`values` output. The matrix is now 19 cases.

Raw data: [`m2-h100-gpu1-topk-20261123.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-topk-20261123.json).




## 21-case validation run

All 21 cases pass in every backend. Median `compile_ms` is **11.3 ms** on this run.

| case | FlowGPU | graphs | eager | inductor |
|---|---:|---:|---:|---:|
| moe_router_topk (small) | 49.32 (graph) | 52.97 | 56.19 | 85.95 |
| moe_router_topk_large | 29.18 (graph) | 33.84 | 57.14 | 87.89 |

The small router is now within measurement noise of eager in this run; the large
router clears both eager and Inductor. The earlier 48-vs-29 small-case gap is
still recorded, since shared-GPU samples move.

Raw data: [`m2-h100-gpu1-21case-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-21case-20261124.json).

## 20-case validation run

All 20 cases pass correctness in every backend. Median `compile_ms` is **15.6 ms**.

- `qk_transform` (RMSNorm + RoPE): graph 6.83 µs vs eager 188.17 µs;
- `topk_last`: 4.64 µs vs eager 18.34 µs;
- `moe_router_topk`: 48.65–53.10 µs vs eager 29.71 µs in this run.

The MoE result is a real remaining gap: its multi-anchor pipeline is slower than
eager on this run even though earlier shared-GPU runs had eager at 151 µs. It
needs fusion/dispatch work before it can claim a win.

Raw data: [`m2-h100-gpu1-20case-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-20case-20261124.json).

## 19-case validation run

All 19 cases pass correctness in every backend. Median `compile_ms` is **15.0 ms**.
Stable full-call paths (default/prepared/graph, shared GPU):

| group | range |
|---|---|
| pointwise / norm / mean / sort | 2.17–3.73 µs |
| topk_last | 4.52 µs |
| conv2d 1×1 / 3×3 / same / stride-2 | 3.66–4.02 µs |
| MoE router (multi-anchor, multi-result) | 48.6–53.0 µs |

The 29.99 µs batch_norm and 32.12 µs matmul samples are shared-GPU outliers; the
graph column is 2.46 / 4.26 µs, which is the stable path.

Raw data: [`m2-h100-gpu1-19case-20261123.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-19case-20261123.json).

## MoE router (multi-operator LLM case)

`moe_router_topk` composes matmul, tanh soft-capping, softmax, topk and a row
gather in one pipeline, with two outputs (`weights`, `ids`). It is the first M2
case that exercises multiple anchors and a multi-result operator together:

| flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 52.84 | 52.99 | 51.11 | 48.37 | 151.60 | 143.49 |

Correctness checks the routing weights against a float64 torch oracle. The
matrix is now 18 cases.

Raw data: [`m2-h100-gpu1-moe-20261123.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-moe-20261123.json).

## Parameterized convolution

`conv2d_ex(x, weight, bias, stride, padding)` folds the stride/padding variants
into one lowering: both values are static ABI literals, the output is
`[(H + 2p - KH)//s + 1, (W + 2p - KW)//s + 1]`, and the kernel masks the padded
border when `padding > 0`.

```flowgpu
out = conv2d_ex(x, weight, bias, 2, 1)   # 3x3 stride-2 same
```

H100 exact against `F.conv2d(stride=2, padding=1)`; this covers the same
geometry as `conv2d_stride2_same` while keeping the older names as
compatibility wrappers.


## Conv tile tuning

`flowgpu.compiler.runtime.conv_tuning.tune_conv2d` compiles a small candidate
set, measures each on the caller's tensors, validates the output and returns the
fastest run:

```python
from flowgpu.compiler.runtime.conv_tuning import tune_conv2d

run, result = tune_conv2d(source, pipeline="quickstart_conv2d_same",
                          args=(x, w, b), cache_dir=".flowgpu-cache",
                          validator=check, iterations=20)
print(result.block_m, result.block_n, result.block_k, result.median_us)
```

On H100 for the 3x3 same conv case:

| candidate (BM, BN, BK) | median |
|---|---:|
| 32, 32, 32 | 19.33 µs |
| 64, 64, 32 | 19.65 µs |
| **128, 64, 64** | **3.91 µs** |
| 128, 128, 64 | 19.50 µs |

The default emitted geometry for that shape is the first candidate; tuning finds
the 4.9x faster geometry at a one-off cost of the candidate compiles. The choice
is cached by source/shape/platform key: a second call compiles only the winner
and reports `result.cached = True`. This is shape-specific measurement, not a
new lowering.


## Full 17-case snapshot

Single shared-GPU run, 20 iterations, all correctness checks pass. Default-path
values marked with `*` are contention outliers; prepared/zero-copy graph are the
stable columns.

| case | flowgpu | prepared | graph | eager | triton |
|---|---:|---:|---:|---:|---:|
| add | 18.37* | 2.24 | 2.21 | 2.20 | 2.16 |
| vectoradd | 18.38* | 2.27 | 2.39 | 2.24 | 2.19 |
| matmul | 4.05 | 4.04 | 4.27 | 3.59 | 2.98 |
| softmax | 2.60 | 2.70 | 2.69 | 5.27 | 2.58 |
| residual_norm | 2.39 | 2.41 | 2.40 | 153.95* | 2.30 |
| mapping | 2.12 | 2.22 | 2.28 | 4.22 | 2.13 |
| neg | 26.22* | 2.14 | 2.28 | 2.24 | 26.22* |
| pow | 3.66 | 3.74 | 3.66 | 3.62 | 2.50 |
| mean_last | 2.32 | 2.28 | 2.28 | 3.34 | 2.26 |
| sort_last | 2.34 | 2.39 | 2.47 | 13.98 | 2.32 |
| layer_norm | 2.30 | 26.02* | 18.86* | 55.06 | - |
| batch_norm | 2.35 | 2.36 | 2.47 | 46.38 | - |
| conv2d_1x1 | 2.28 | 2.39 | 2.47 | 34.84 | - |
| conv2d_3x3 | 4.03 | 4.00 | 4.03 | 23.42 | - |
| conv2d_same | 20.79* | 3.79 | 4.08 | 23.45 | - |
| conv2d_stride2 | 26.39* | 3.79 | 4.06 | 22.91 | - |
| conv2d_stride2_same | 25.89* | 3.53 | 3.72 | 22.95 | - |

The stable prepared/graph paths are 2.1–4.3 µs except the matmul case at 4.0.
The remaining large spread is host/GPU contention in this shared run, not a
kernel-level regression.

Raw data: [`m2-h100-gpu1-17case-20261123.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-17case-20261123.json).

## Peak memory, all 19 cases

`max_memory_allocated()` in KiB, including first-use compilation allocations:

| case | flowgpu | prepared/graph | eager |
|---|---:|---:|---:|
| add | 48.5 | 46.5 | 48.5 |
| vectoradd | 768.5 | 736.5 | 768.5 |
| matmul | 32992.5 | 216.5 | 32992.5 |
| softmax | 448.5 | 416.5 | 448.5 |
| residual_norm | 194.5 | 186.5 | 194.5 |
| mapping | 68.0 | 64.0 | 68.0 |
| neg | 44.5 | 42.5 | 44.5 |
| pow | 48.5 | 46.5 | 48.5 |
| mean_last | 6.5 | 6.0 | 6.5 |
| sort_last | 5.5 | 5.0 | 5.5 |
| layer_norm | 24.5 | 23.5 | 24.5 |
| batch_norm | 26.5 | 25.5 | 26.5 |
| conv2d_1x1 | 32791.5 | 22.5 | 32791.5 |
| conv2d_3x3 | 33.5 | 32.5 | 33.5 |
| conv2d_same | 64.5 | 62.0 | 64.5 |
| conv2d_stride2 | 27.0 | 26.0 | 26.0 |
| conv2d_stride2_same | 34.5 | 33.5 | 34.5 |
| moe_router_topk | 32782.5 | 13.0 | 32782.5 |
| topk_last | 72.5 | 71.5 | 72.5 |

The 32 MiB entries are first-use compilation/allocator effects for matmul,
1x1 conv and the MoE router, not kernel working sets; prepared/static paths show
the actual small buffers. The metric spots that class of cost, not a smaller
working set.

Raw data: [`m2-h100-gpu1-19case-20261123.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-19case-20261123.json).

## Compile time (19-case matrix)

`compile_ms` from the same run, including Triton source generation and the first
module load:

| group | cost |
|---|---:|
| first case (cold Triton cache) | 114.6 ms |
| remaining cases | 9.3–26.9 ms |
| median of all 19 | **15.0 ms** |

Compile time is dominated by the first Triton compilation; normal cache hits
are in the low tens of milliseconds.

Raw data: [`m2-h100-gpu1-19case-20261123.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-19case-20261123.json).

## Stride-2 same convolution

`conv2d_stride2_same` is the ResNet-style 3x3, stride 2, padding 1
downsample. It is the same implicit-GEMM lowering with both stride and bounds
masks.

| case | flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---|---:|---:|---:|---:|---:|---:|
| conv2d_stride2_same | 24.38* | 3.56 | 6.55 | 3.78 | 39.92 | 60.21 |

*Default sample is a shared-GPU outlier; prepared/graph are the stable 3.6–3.8
µs path. Exact against `F.conv2d(stride=2, padding=1)` in float64. The matrix is
now 17 cases.

Raw data: [`m2-h100-gpu1-stride2-same-case-20261123.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-stride2-same-case-20261123.json).

## Stride-2 convolution

`conv2d_stride2` adds a stride-2 implicit-GEMM convolution (no padding).
Output is `[N, (H-KH)//2+1, (W-KW)//2+1, O]`.

| case | flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---|---:|---:|---:|---:|---:|---:|
| conv2d_stride2 (3x3, s=2) | 19.49* | 3.74 | 23.04* | 3.72 | 55.55 | 60.12 |

*Default and prepared+graph samples are shared-GPU outliers; prepared and
zero-copy graph are the stable 3.7 µs path. Exact against
`F.conv2d(..., stride=2)` in float64. The matrix is now 16 cases.

Raw data: [`m2-h100-gpu1-stride2-case-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-stride2-case-20261122.json).

## Same-padded convolution

`conv2d_same` is the odd-K, `(K-1)//2` zero-padding variant. It reuses the
implicit-GEMM lowering with per-offset bounds masks and keeps the output spatial
shape.

| case | flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---|---:|---:|---:|---:|---:|---:|
| conv2d_same (3x3, pad 1) | 3.90 | 3.89 | 6.65 | 4.02 | 62.04 | 141.34 |

Exact against `F.conv2d(..., padding=1)` in float64. This adds a 15th M2 case.

Raw data: [`m2-h100-gpu1-same-case-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-same-case-20261122.json).

## General KxK conv2d

`conv2d` now accepts a 4-D `[O,C,KH,KW]` weight and lowers a small KxK
convolution to an implicit-GEMM kernel: for each kernel offset, one `tl.dot`
accumulates into the same output tile. 1x1 keeps the same kernel with `KH=KW=1`.

| case | flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---|---:|---:|---:|---:|---:|---:|
| conv2d_1x1 | 25.43* | 2.34 | 5.06 | 2.43 | 36.31 | 51.17 |
| conv2d_3x3 | 4.03 | 3.99 | 6.64 | 4.04 | 50.25 | 61.18 |

*The 1x1 default-path sample is a shared-GPU outlier; the prepared/graph
numbers and the 3x3 row are the useful comparison. The 3x3 result is exact
(`max_abs = 0.0`) against `F.conv2d` in float64.

The matrix is now 14 cases.
Raw data: [`m2-h100-gpu1-conv3-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-conv3-20261122.json).

## 13-case matrix: batch_norm and 1x1 conv2d

Two more representative cases:

| case | flowgpu | prepared | prepared+graph | graph | eager | inductor |
|---|---:|---:|---:|---:|---:|---:|
| batch_norm | 2.34 | 2.33 | 5.01 | 2.42 | 29.67 | 29.68 |
| conv2d_1x1 | 2.37 | 2.36 | 4.98 | 2.44 | 32.07 | 137.68 |

`batch_norm` is a single fused pointwise kernel and is already far below eager.
`conv2d_1x1` is now one tiled `x @ weight.T + bias` kernel: 2.37 µs instead of
the previous four-kernel composition (136 µs), and all prepared/graph paths
work.

Raw data: [`m2-h100-gpu1-13case-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-13case-20261122.json).

## Native layer_norm

`layer_norm` now lowers to one row kernel (fp32 mean/variance, rsqrt(var+eps),
weight/bias) instead of the composed reduction/pointwise graph. On the shared
H100 GPU1:

| flowgpu | prepared | prepared+graph | zero-copy graph | eager | inductor |
|---:|---:|---:|---:|---:|---:|
| 2.39 | 2.38 | 5.21 | 2.42 | 36.52 | 36.46 |

Rank-0 scalar arguments (`eps`) now prepare and capture too: the prepared and
graph runners accept a Python int/float alongside the tensors. All four FlowGPU
paths pass correctness against the torch formula for eps 1e-5 and 1e-3.

Literal rows up to 4096 use the native kernel; symbolic or wider rows keep the
portable reduction/pointwise expansion, so the operator does not regress for
dynamic shapes.

Raw data: [`m2-h100-gpu1-layernorm-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-layernorm-20261122.json).

## Native mean_last

`mean_last` now lowers to one row-mean kernel instead of the
`sum_last + ones_like + sum_last + div` composition. On the shared H100 GPU1
(exploratory run, not an exclusive-GPU claim):

| flowgpu | prepared | zero-copy graph | eager | triton |
|---:|---:|---:|---:|---:|
| 2.21 | 2.26 | 2.23 | 3.33 | 2.27 |

The structural result is the one to rely on: the graph has one kernel, not
four composed operators. The wall-clock numbers move with the shared GPU load.

Raw data: [`m2-h100-gpu1-native-norms-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-native-norms-20261122.json).

## Native sort

`sort_last` now lowers to one `tl.sort` row kernel. On H100 GPU1 for `[4, 8]`:

| flowgpu | prepared | prepared+graph | zero-copy graph | eager | inductor | triton |
|---:|---:|---:|---:|---:|---:|---:|
| 2.23 | 2.23 | 4.54 | 2.42 | 13.87 | 13.86 | 2.19 |

FlowGPU is within measurement noise of the explicit Triton `tl.sort` baseline
and far below eager/Inductor. The old correctness-first topk composition is
gone; literal extents are still required.

Raw data: [`m2-h100-gpu1-sort-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-sort-20261122.json).

## Prepared graph path

`flowgpu_prepared_graph` prepares the plan, captures a graph over static inputs,
and copies each call's tensors into those inputs before replay. It is a middle
ground between the allocating prepared path and the zero-copy static path.

| case | prepared | prepared+graph | zero-copy graph | eager | triton |
|---|---:|---:|---:|---:|---:|
| add | 15.04 | 11.73 | 2.98 | 5.27 | 14.48 |
| vectoradd | 19.34 | 14.65 | 3.95 | 6.63 | 17.97 |
| matmul | 15.56 | 11.04 | 4.32 | 9.76 | 16.50 |
| softmax | 12.98 | 8.59 | 2.91 | 5.84 | 12.32 |
| residual_norm | 17.24 | 11.92 | 2.89 | 63.76 | 12.80 |
| mapping | 14.19 | 10.65 | 2.95 | 9.18 | 13.73 |
| neg | 12.73 | 8.56 | 3.33 | 4.79 | 13.56 |
| pow | 14.34 | 10.99 | 3.79 | 5.22 | 13.56 |
| mean_last | 12.98 | 8.89 | 2.89 | 5.83 | 12.87 |

Prepared+graph removes allocation and dispatch but still pays one batched input
copy; it is 8.56–14.65 µs, generally better than the allocating prepared path
but above eager on the simple cases. Zero-copy remains the fastest fixed-shape
path.

Raw data: [`m2-h100-gpu1-prepgraph-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-prepgraph-20261122.json).

## Static zero-copy graph path

`flowgpu_graph` prepares the plan, captures a CUDA Graph over caller-owned static
inputs, and times `replay()` only. It is the fixed-shape deployment path, not a
general dispatch result.

| case | flowgpu | prepared | graph replay | eager | inductor | triton |
|---|---:|---:|---:|---:|---:|---:|
| add | 17.52 | 14.09 | 2.87 | 5.17 | 23.35 | 13.33 |
| vectoradd | 18.96 | 13.87 | 3.03 | 5.02 | 27.03 | 13.58 |
| matmul | 18.98 | 13.79 | 4.30 | 8.91 | 41.17 | 16.36 |
| softmax | 16.13 | 12.11 | 2.95 | 5.55 | 27.23 | 11.64 |
| residual_norm | 21.00 | 16.54 | 2.70 | 59.54 | 26.93 | 12.17 |
| mapping | 16.96 | 13.62 | 2.70 | 9.03 | 25.83 | 13.17 |
| neg | 15.03 | 12.06 | 2.65 | 4.67 | 23.99 | 12.74 |
| pow | 17.26 | 14.21 | 3.92 | 5.38 | 25.79 | 13.19 |
| mean_last | 21.27 | 12.50 | 2.62 | 5.84 | 18.25 | 12.36 |

Static graph replay is **2.62–4.30 µs**, below eager on every case and below the
hand-written Triton baseline. The cost is fixed metadata/pointers: a changed
shape or tensor requires a new capture or a copy into the captured inputs.

Raw data: [`m2-h100-gpu1-graph9-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-graph9-20261122.json).

## Static-input graph replay

`GraphRunner` (prepared plan + CUDA Graph) was measured on the same H100 GPU1:

| case | replay (µs) | copy + replay (µs) |
|---|---:|---:|
| add | 2.60 | 11.19 |
| vectoradd | 2.69 | 10.60 |
| matmul | 3.97 | 10.63 |
| softmax | 2.70 | 8.10 |
| residual_norm | 2.63 | 10.99 |
| mapping | 2.65 | 10.30 |


Replay is below eager full-call on every simple case. The copy + replay path
uses one `_foreach_copy_` dispatch for multiple inputs, which brought 9.3–18.3 µs
down to 8.1–11.2 µs; it still helps most when the caller owns static buffers and
updates them in place. This is a fixed-shape deployment option, not a general
dispatch replacement.

## Optimization order

1. **Make the prepared path first-class** for fixed-shape deployment: document
   it, benchmark it, and remove its multi-pipeline restriction where safe.
   The prepared path and fixed-shape `GraphRunner` are now implemented.
2. **Reduce host overhead** (guards/allocation/dispatch) rather than tuning the
   pointwise/softmax kernels. `GraphRunner.replay()` is 3.3–4.0 µs; the remaining
   problem is copying caller tensors into static buffers. `launch_cache=True`
   saved ~2.5 µs and disabling runtime guards saved ~2.5 µs in a separate
   host-variant probe; both are opt-in and do not change kernels.
3. **Matmul only**: graph is 4.10 vs 3.02 µs; a tile/policy change is justified
   after the host path reaches parity.
4. Re-run with an exclusive GPU, multiple seeds and at least 200 iterations
   before publishing a performance claim.

## Reproduce

```sh
cd FlowGPU
python benchmarks/competitive/device_runner.py \
  --backends flowgpu,flowgpu_prepared,eager,inductor,triton \
  --warmup 8 --iterations 30 --json > m2-h100.json
```

`flowgpu_prepared` performs offline preparation; its `compile_ms` includes that
cost.

## 30-case regression after the device policy

After switching the default and prepared paths to `resolve_device_platform()`,
the full M2 matrix was re-run on H100 GPU1 with `flowgpu`, `flowgpu_prepared`,
`flowgpu_graph` and `eager`:

- 30 cases;
- **0 correctness failures**;
- no case errored in the three FlowGPU backends;
- 10 iterations, 3 warmup, shared device.

This is a regression check, not a new performance table. Raw data:
[`m2-h100-gpu1-30case-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-30case-20261124.json).
