# What do the generated guards cost?

Same subgraph, same weights, same dedicated GPU0; only
`runtime_validation` changes. Seven randomized interleaved rounds x 30
iterations (0% GPU0 utilization before and after).

| tokens | guarded / unguarded | guarded | unguarded |
|---:|---:|---:|---:|
| 1 | **1.018x** (1.013–1.025) | 227.6 us | 223.5 us |
| 2048 | 1.003x (0.89–1.05) | 1761 us | 1756 us |

## Interpretation

- At decode, generated input-contract guards cost about **1.8%** — the
  auditability tax is small on the launch-bound path.
- At prefill, the guard block is within measurement noise; the kernel time
  dominates. The wide range is a reminder that even dedicated H100 runs have
  session noise.
- This measures **host guards only**, not compile-time checks or the CPU
  reference. The full auditability cost includes tests and oracle runs.

This is the answer to "does auditability require a large runtime tax?" on this
subgraph: not at decode; prefill inconclusive but bounded.

Raw data:
[`guard-cost-h100-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/guard-cost-h100-20261125.json).
