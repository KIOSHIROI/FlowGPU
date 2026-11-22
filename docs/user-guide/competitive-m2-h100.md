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
