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

## Optimization order

1. **Make the prepared path first-class** for fixed-shape deployment: document
   it, benchmark it, and remove its multi-pipeline restriction where safe.
2. **Reduce host overhead** (guards/allocation/dispatch) rather than tuning the
   pointwise/softmax kernels. `launch_cache=True` saved ~2.5 µs and disabling
   runtime guards saved ~2.5 µs in a separate host-variant probe; both are
   opt-in and do not change kernels.
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
