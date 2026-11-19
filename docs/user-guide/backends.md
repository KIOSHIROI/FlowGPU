# Backends and platform support

FlowGPU has two backend families:

| Backend | Use | Requirements |
|---|---|---|
| Reference | semantic oracle, CPU execution, tests | Python only |
| Triton | compiled operator kernels | PyTorch + Triton + a supported device |

`compile_source` emits **both** by default, so you can inspect the Triton module
and the reference implementation side by side.

## Reference backend

The reference backend is portable Python emitted from the same typed dataflow
graph. It is used for:

- correctness tests;
- `--run` on any machine;
- comparison with generated Triton on a device.

It is not optimized; it is the semantic oracle.

## Triton backend

The Triton backend emits:

- `@triton.jit` kernels;
- a host wrapper with allocations, shape/dtype validation and launch arguments;
- explicit refusals when a construct cannot be lowered exactly.

On CUDA, `--run-gpu` executes the generated module and compares it with the
reference backend.

## Platform profiles

`--platform` selects a target profile. Profiles differ in measured backend
contracts, legal kernel kinds, index width, tiling and launch limits.

| Profile | Status |
|---|---|
| `cuda` | default; CUDA-class devices |
| `cuda-tuned` | measured GEMM/tiling policy for supported CUDA shapes |
| `ascend` / `ascend-910b` / `ascend-wide` | experimental; measured contracts, fail-closed gaps |
| `kunlunxin` / `kunlun-r300` | experimental; measured contracts, fail-closed gaps |

Non-CUDA profiles are not a promise of correctness for every program. They are
contract-driven experiments; if a construct is not validated, the compiler
refuses it rather than silently miscompiling it.

### Layout views

On CUDA, layout-view fusion is enabled by default. On Ascend/KunlunXin it is
disabled until the relevant indexed-layout contracts are device-validated; an
explicit `layout_views=True` on those profiles fails closed. Use
`layout_views=False` to force materialisation.

## Generated vs. hand-written code

`flowgpu/flowgpu/compiler/extern.py` lets a pipeline record a hand-written Triton kernel as an
extern. The evidence/tooling keeps generated and extern kernels separate; see
[`docs/EXTERN_LEDGER.md`](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/EXTERN_LEDGER.md).

## Further reading

- [Backend contracts](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/BACKEND_CONTRACTS_20261019.md)
- [Support matrix](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/SUPPORT_MATRIX.md)
- [Device layout](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/DEVICE_LAYOUT_20261027.md)
