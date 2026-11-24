# FlagOS / FlagGems baseline

The public matrix now includes a **FlagOS/FlagGems** column for the operators
that run on this stack. This is an external baseline, not a FlowGPU improvement.

## Environment

- FlagGems `5.0.2` (`pip install --no-deps flag_gems`, plus `packaging`,
  `pyyaml`, `sqlalchemy`)
- PyTorch `2.14.1+cu130`, Triton `3.8.0`
- NVIDIA H100 GPU1, shared; 20 iterations

## Measured (full-call µs)

| case | FlowGPU | FlagGems | eager | Inductor |
|---|---:|---:|---:|---:|
| softmax | 17.09 | 19.07 | 5.79 | 22.47 |
| mean_last | 15.23 | 19.10 | 6.17 | 26.33 |
| sort_last | 14.31 | 174.93 | 23.94 | 70.53 |
| topk_last | 16.88 | 43.42 | 18.77 | 27.66 |
| layer_norm | 17.24 | 31.11 | 72.82 | 26.47 |
| conv2d_1x1 | 18.21 | 44.94 | 45.79 | 68.44 |
| conv2d_3x3 | 18.86 | 44.35 | 56.30 | 86.78 |
| conv2d_same | 18.29 | 44.50 | 57.94 | 85.83 |
| conv2d_stride2 | 18.76 | 45.55 | 57.96 | 87.87 |
| conv2d_stride2_same | 18.17 | 44.48 | 57.71 | 90.29 |

All correctness checks pass. FlagGems `sort` is much slower here; flowgpu's
native sort is the better path. FlowGPU is ahead on norm/conv/topk, close on
softmax/mean.

Excluded:

- `matmul` — FlagGems 5.0.2 aborts with this Triton version in
  `make_tensordesc` specialization;
- `add`/pointwise — the BlockPtr path fails to compile with Triton 3.8;
- batch_norm — unsupported language construct;
- MoE/QK/router composites — no single FlagGems equivalent.

## TVM

`apache-tvm 0.27.0.post1` installs but the published wheel reports
`tvm.runtime.enabled("cuda") == False`; it is CPU-only. A GPU-comparable TVM
baseline needs a source build with CUDA, which is not present in this
environment. No TVM number is published rather than publishing a CPU number as
if it were a GPU baseline.

Raw data: [`m2-h100-gpu1-flagos-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/m2-h100-gpu1-flagos-20261124.json).
