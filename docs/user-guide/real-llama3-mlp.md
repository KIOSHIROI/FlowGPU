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

## Why this task matters

It exposes the real boundary of the current implementation:

- correctness and integration work on a real model subgraph;
- the remaining gap is GEMM tile/scheduling quality, not the language;
- next steps are tile/autotune, split-K for decode, and gate/up packing.

Raw ratio data: [`llama3-mlp-h100-gpu1-20261124.json`](https://github.com/KIOSHIROI/FlowGPU/blob/main/benchmarks/competitive/measurements/llama3-mlp-h100-gpu1-20261124.json).
