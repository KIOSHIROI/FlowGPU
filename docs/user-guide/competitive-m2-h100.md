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



## Extended operator matrix

The same runner now also covers the newly added `neg`, `pow` and `mean_last`
operators. All 9 cases pass correctness in all five backends. Full-call time
(µs, H100 GPU1, 30 iterations):

| case | flowgpu | prepared | eager | inductor | triton |
|---|---:|---:|---:|---:|---:|
| neg_literal | 16.06 | 12.98 | 4.67 | 24.23 | 14.46 |
| pow_literal | 17.71 | 14.28 | 5.16 | 25.66 | 13.88 |
| mean_last | 21.70 | 12.89 | 5.77 | 18.05 | 13.03 |

Prepared `mean_last` is 12.89 µs, faster than the composed default and close to
the hand-written Triton reduction (13.03). The default 21.70 µs is the cost of
the extra `sum_last(ones_like(x))` count reduction; folding that count is the
next optimization.

Raw data: [`m2-h100-gpu1-20261122-extended.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-20261122-extended.json).


## Multi-seed medians

Three seeds (0, 1, 2), 30 iterations each; values are medians of the per-seed
full-call times. All correctness checks pass for every seed.

| case | flowgpu | prepared | eager | inductor | triton |
|---|---:|---:|---:|---:|---:|
| add_literal | 17.81 | 14.49 | 4.99 | 23.08 | 13.72 |
| vectoradd_symbolic | 19.21 | 14.44 | 5.23 | 26.30 | 14.00 |
| matmul_symbolic | 18.54 | 13.91 | 9.16 | 41.92 | 16.11 |
| softmax_wide | 16.38 | 12.12 | 5.36 | 26.42 | 11.72 |
| residual_norm | 21.53 | 16.54 | 60.37 | 25.98 | 12.42 |
| mapping_pointwise | 17.08 | 13.97 | 9.44 | 25.13 | 13.29 |
| neg_literal | 15.24 | 12.44 | 4.77 | 23.99 | 13.60 |
| pow_literal | 17.30 | 14.19 | 5.07 | 25.00 | 13.55 |
| mean_last | 21.94 | 12.80 | 5.74 | 17.71 | 12.12 |

The single-seed conclusions hold: prepared is close to explicit Triton, eager
still wins the simple pointwise/reduction cases, and the default `mean_last`
carries the extra count reduction.

Raw data: [`m2-h100-gpu1-multiseed-20261122.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-multiseed-20261122.json).

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
