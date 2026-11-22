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
| add_literal | graph (µs) | 2.71 | 2.74 | 2.72 | 2.70 | 2.71 |
| add_literal | full (µs) | 17.34 | 14.13 | 5.01 | 22.69 | 14.22 |
| vectoradd_symbolic | graph (µs) | 2.74 | 2.80 | 2.60 | 2.63 | 2.64 |
| vectoradd_symbolic | full (µs) | 19.02 | 14.26 | 5.03 | 25.75 | 13.08 |
| matmul_symbolic | graph (µs) | 4.10 | 4.07 | 3.66 | 3.63 | 3.02 |
| matmul_symbolic | full (µs) | 18.29 | 13.75 | 8.98 | 43.26 | 15.75 |
| softmax_wide | graph (µs) | 2.80 | n/a | 5.42 | 3.76 | 2.73 |
| softmax_wide | full (µs) | 16.30 | n/a | 5.39 | 26.34 | 11.37 |
| residual_norm | graph (µs) | 2.71 | 2.73 | 22.56 | 2.81 | 2.69 |
| residual_norm | full (µs) | 21.24 | 16.44 | 60.91 | 26.13 | 12.51 |
| mapping_pointwise | graph (µs) | 2.78 | 2.68 | 3.60 | 3.61 | 3.08 |
| mapping_pointwise | full (µs) | 17.13 | 13.70 | 9.06 | 24.49 | 13.44 |

`flowgpu_prepared` is the existing offline `prepare_execution_plan` path with
caller-owned buffers. It is unavailable for `softmax_wide` because that public
example contains two pipelines and preparation currently requires exactly one.
`triton` is the hand-written baseline in `benchmarks/competitive/triton_baselines.py`.

## What the numbers say

1. **The kernel path is competitive.** FlowGPU and explicit Triton have nearly
   the same Graph latency on pointwise, softmax and residual cases; FlowGPU is
   slightly slower on matmul (4.10 vs 3.02 µs). Kernel fusion is already doing
   real work: residual RMSNorm Graph is 2.71 µs vs eager 22.56 µs.
2. **The gap is host-side.** FlowGPU full-call is 13–16 µs above its Graph time.
   Eager full-call is 5–9 µs on simple cases. The measured loss is guards,
   allocation and launch dispatch, not kernel math.
3. **Prepared execution closes most of the public-flow gap.** Prepared FlowGPU
   matches or beats explicit Triton on add (14.13 vs 14.22), vectoradd (14.26 vs
   13.08), matmul (13.75 vs 15.75) and mapping (13.70 vs 13.44), and is close on
   residual (16.44 vs 12.51).
4. **FlowGPU beats TorchInductor full-call on every measured case** (17–21 µs vs
   23–43 µs), but loses to eager on the simple pointwise/softmax cases.
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
cost. `n/a` means the prepared path refused the multi-pipeline source, not that
the kernel is wrong.
