# Device-policy contribution on one H100

This isolates the tile policy from the rest of the compiler. Same subgraph,
same weights, same device; only the platform config changes:

- **safe** (`resolve_platform("cuda")`): 128x128 large-N, 16x64 small-M;
- **tuned** (`resolve_device_platform()`): 128x256 large-N, 32x64x128 small-M.

Dedicated GPU0 (0% utilization before and after; GPU1 busy), 3 randomized
interleaved rounds x 15 iterations.

| tokens | safe / tuned | tuned / eager | safe / eager |
|---:|---:|---:|---:|
| 1 | 1.027x | 1.009x | 1.036x |
| 2048 | **1.163x** | 1.063x | 1.237x |

## Interpretation

- The device policy is worth **2.7% at decode** and **16.3% at prefill** on
  this H100; the safe L4 policy is leaving prefill performance on the table.
- The `tuned / eager` values are not a cuBLAS claim: this run has random
  weights and one session. The **safe / tuned** ratio is the controlled
  quantity because both sides share everything except the tile policy.
- This does not prove L4 behavior. It quantifies why a device-aware policy is
  needed, and what a single policy would cost on H100.

Raw data:
[`device-policy-h100-20261125.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/device-policy-h100-20261125.json).
