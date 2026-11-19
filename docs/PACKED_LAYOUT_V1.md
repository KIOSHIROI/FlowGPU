# Packed layout v1: descriptor, codecs, and the first extern migration (2026-10-27)

This is the implementation slice of `docs/PACKED_LAYOUT_DESIGN.md`: enough to
state **MXFP8** in the language and to delete one hand-written extern kernel,
without growing the frontend grammar.

## What exists

| Piece | Where | State |
|---|---|---|
| Layout descriptor | `flowgpu/compiler/packed_layout.py::PackedLayout` | data-only: `storage=e4m3`, `block=32`, `scale=e8m0`, `scale_dtype=uint8`, `pack_bits=8`; other values are rejected rather than half-supported |
| E4M3 byte codec (RTNE, saturating) | `packed_layout.e4m3_byte` / `e4m3_value` | independent bit-layout oracle; agrees byte-for-byte with `tools/fp8_encoder.e4m3_bits` and numerically with the backend's `fp8e4m3` preamble |
| E8M0 scale codec | `packed_layout.e8m0_scale` / `e8m0_value` | `ceil(log2(amax/448))+127`, clamped `[0,254]`, amax clamped to `1e-30` |
| MXFP8 quantise/dequantise oracle | `quantize_mxfp8` / `dequantize_mxfp8` | no torch, no device; this is the statement's semantics |
| DSL program | `examples/packed/mxfp8_quantize.fgpu` | 19 frontend operators; compiles to **one** `primitive` Triton kernel plus a runnable reference module |
| Agreement test | `test/test_packed_layout.py` | the DSL program's reference output matches the oracle **byte-for-byte** for both data and scale bytes |

The language needs no new operator: the value layer already provides
`float_bits`/`bits_float`, bitwise ops on `int32`, `cast_to(fp8e4m3)` and
`cast_to(uint8)`.  Integer constants are program inputs because the value layer
mixes scalars with tensors only through ordinary arithmetic.

## The first migration: `to_mxfp8`

FlagOS task 174's `to_mxfp8` was a hand-written three-pass Triton kernel
(`_block_amax`, `_block_scale`, `_block_encode`) with a separate KunlunXin
variant.  The DSL program states the same computation as one fused kernel and is
pinned against the same block/rounding rules, so the extern kernel no longer
needs a special backend path:

| | hand-written extern | DSL program |
|---|---|---|
| kernels | 3 passes (amax, scale, encode) | 1 fused `primitive` kernel |
| platforms | generic + a KunlunXin variant for a bitcast miscompile | target-neutral source; the backend plan chooses the machine code |
| verification | per-platform harness runs | reference oracle + byte-exact test in the normal suite |
| provenance | extern envelope (`flowgpu/compiler/extern.py`) | frontend operator graph |

Status: **reference-verified**; the Triton path compiles (`complete`, one kernel,
`fp8e4m3`/`uint8` outputs) and is expected to run on CUDA, but no GPU was
available in this environment — the nightly `tools/gpu_smoke.py` workflow is
where that gets pinned.  The extern file itself stays in
`private evidence` as the historical submission; the
ledger (`docs/EXTERN_LEDGER.md`) records it as promoted.

## Boundaries and next steps

- **4-bit packing is not in v1.** `PackedLayout` rejects `pack_bits=4`; E2M1 +
  UE8M0 (FlagOS task 172 / CANN fp4) is the next descriptor extension, and it is
  the reason `pack_bits` is a field instead of a constant.
- **E5M2 and per-row/per-tensor scales** are not in v1; the descriptor validation
  refuses them instead of silently applying the E4M3 rule.
- **The descriptor is metadata, not a type.** A follow-up can attach it to
  pipeline input/output declarations (`input q : packed<...>`) once there is a
  second concrete consumer; doing it earlier would freeze a syntax no test needs.
- **Device verification** of the generated MXFP8 module belongs to the GPU smoke
  job and to a CANN submission if we choose to re-enter 174 with the DSL version.
