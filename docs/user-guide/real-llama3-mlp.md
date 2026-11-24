# Real task: Llama 3 8B SwiGLU MLP

`examples/llama3_mlp_swiglu.fgpu` is one real hot subgraph from Llama 3 8B:

```text
x: [T, 4096]
w_gate, w_up: [4096, 14336]
w_down: [14336, 4096]
```

It is the MLP block of every decoder layer. The host keeps attention, KV cache,
batching and sampling; FlowGPU compiles only this block.

## What FlowGPU generates

Three kernels:

- `k0`: gate matmul;
- `k1`: up matmul with `silu(gate) * up` fused into its epilogue;
- `k2`: down matmul.

The `hidden` tensor is `[T, 14336]` and is the buffer between `k1` and `k2`.
`T` is 2048 for prefill and 1 for decode in the measurements below.

## Correctness

Against PyTorch eager bf16, the relative maximum error is:

| tokens | relative error |
|---:|---:|
| 1 | 3.2e-3 |
| 2048 | 0 (same bf16 rounding) |

The correctness protocol uses a scaled slice (`T=2, H=8, I=16`) for the CPU
reference; the full shape is checked against the PyTorch GPU oracle.

## Performance (H100 PCIe, shared, bf16)

| tokens | FlowGPU / eager | Graph / eager |
|---:|---:|---:|
| 1 (decode) | **1.02x** | 1.20x |
| 2048 (prefill) | **1.02x** | 1.05x |

The latest 3-repeat measurement is about **2% slower than cuBLAS**. Earlier
sessions on the same shared GPU measured 1.08–1.18x; absolute times moved with
device load, so only ratios are reported. This is not a speed claim: FlowGPU is
at parity, not ahead.

The graph path is slower here because this workload is compute-bound; CUDA
Graph removes launch overhead, not GEMM time.


## Opt-in H100 tile policy

`resolve_device_platform()` selects `128x256x64, 3 stages, 8 warps` when the
device exposes at least 150 KiB of opt-in shared memory (H100 class). The
default stays at the L4-safe 128x128x64 policy; the large tile needs about
144 KiB and is not safe on L4-class devices.

```python
from flowgpu.compiler.backend.platform_configs import resolve_device_platform
run = compile_pipeline(..., platform_config=resolve_device_platform())
```

On H100 the large tile brings prefill to about 1.00–1.02x eager, versus
1.05x for the default tile in the same session. This is a device policy, not a
new global default: L4 and other 100 KiB-class devices keep the safe tile.


## Prepared plans select the device tile automatically

`prepare_execution_plan()` now resolves the GEMM policy from the actual device
when no `platform_config` is passed. Prepared plans therefore use the H100
128x256 tile and keep the L4-safe 128x128 tile elsewhere.

In one 3-repeat session on a shared H100, the prepared plan measured:

| tokens | prepared / eager | graph / eager | default / eager |
|---:|---:|---:|---:|
| 1 | 1.04x | 1.05x | 1.03x |
| 2048 | **0.94x** | 0.97x | 1.11x |

The prefill prepared path reused output buffers and the larger tile; decode is
still about 4% slower. Shared-device load moved absolute times, so treat this
as a deployment-path observation, not a universal speed claim.

## Why this task matters

It exposes the real boundary of the current implementation:

- correctness and integration work on a real model subgraph;
- the remaining gap is GEMM tile/scheduling quality, not the language;
- next steps are tile/autotune, split-K for decode, and gate/up packing.

Raw ratio data: [`llama3-mlp-h100-gpu1-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/llama3-mlp-h100-gpu1-20261124.json), [`device-tile`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/llama3-mlp-h100-gpu1-device-tile-20261124.json).
