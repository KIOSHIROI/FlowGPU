# Competitive roadmap

FlowGPU is not trying to replace TVM/XLA/Triton on general compiler coverage.
It is trying to become a **contract-gated, auditable layer for portable LLM
inference kernels**: compile a small tensor program, prove which target can
legally run the chosen form, freeze the measured configuration, and ship the
evidence with the kernel.

## Competitor map

| Area | Representative systems | FlowGPU position |
|---|---|---|
| general tensor/graph compilers | TVM/Relax, XLA/OpenXLA, IREE/MLIR, TorchInductor | not competitive on breadth; integrate, do not replace |
| kernel DSL/codegen | Triton, TileLang, Mojo, CUTLASS/CuTe | FlowGPU is a consumer/peer, not a replacement |
| cross-chip stacks | FlagOS/FlagTree, Triton-Ascend, FlagGems | direct competitor; FlowGPU's bet is measured legality contracts, not operator count |
| end-to-end LLM serving | TensorRT-LLM, vLLM, SGLang, MLC-LLM | out of scope; FlowGPU targets the operator layer |
| superoptimization | TASO, Tensat, egg/Souper | CCSA is a research prototype, not a product claim |
| constrained/trustworthy codegen | backend contracts + evidence bundles | FlowGPU's narrow differentiation |

## What "catch up" means here

We will track four separate axes instead of one leaderboard number:

1. **Correctness/legality**: cases compile and agree with an independent oracle;
2. **Coverage**: number of representative LLM-inference operators expressible;
3. **Performance**: device latency vs PyTorch eager, `torch.compile` and a
   pinned Triton baseline on identical inputs, tolerances and timing;
4. **Deployment**: frozen autotune, per-shape cache, package install, docs.

## Milestones

| milestone | deliverable | evidence |
|---|---|---|
| M0 | CPU competitive probe over six public cases | `benchmarks/competitive/probe.py` |
| M1 | correctness oracle for the same cases | reference vs PyTorch, max error per dtype |
| M2 | device latency runner | FlowGPU Triton vs eager/Inductor/Triton, Graph and full-call |
| M3 | dynamic-shape specialization | hit rate, rebuild cost, guarded cache |
| M4 | coverage expansion | new operators with independent oracle + tests |
| M5 | release/adoption | PyPI, docs, reproducible benchmark report |

## Non-goals

- claiming a general-compiler win over TVM/XLA;
- tuning to a single benchmark shape or a single chip;
- reporting a structural probe as a latency result;
- publishing private competition or evidence data.

## Current status

M0 is available as a reproducible structural probe over six public cases. The
probe compiles all six and reports compile time, kernels, slots and traffic, not
device latency.

M1 is also available for the same six cases: the FlowGPU reference backend is
checked against independent flat-list oracles (add, int pointwise, matmul,
softmax, residual RMSNorm). All six pass on the development machine. The
residual-Norm case uses a small tolerance for bf16 rounding; per-case tolerances
are declared in `cases.json`.

M3's dispatch primitive is implemented as `runtime.dynamic_shape.DynamicShapeRunner`:
it derives symbolic bindings from the actual tensors and caches a static
specialization per metadata tuple, with explicit rebuild on shape change.

M2's runner is implemented as `benchmarks/competitive/device_runner.py`:
same inputs, float64 oracle and per-case tolerance, Graph and full-call
separated, FlowGPU vs eager vs `torch.compile` (TorchInductor). It is not claimed
as a result yet because the development host has no CUDA device; CI validates
only the matrix configuration.

Run it on a CUDA host with:

```sh
python benchmarks/competitive/device_runner.py --json
```
