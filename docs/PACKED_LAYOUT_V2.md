# Packed layout v2: NVFP4 and AWQ int4 (2026-10-27)

v1 (`docs/PACKED_LAYOUT_V1.md`) covered MXFP8: E4M3 values with one E8M0 scale per
32-value block.  v2 adds the two 4-bit layouts the extern ledger lists as
candidates, with **byte-exact reference oracles** and independent statement
translations in the test suite.

## What exists

| Layout | Descriptor | Values | Scale | Packing | Evidence |
|---|---|---|---|---|---|
| MXFP8 (v1) | `MXFP8` | E4M3 | E8M0 per block 32 | 1 value/byte | unchanged |
| **NVFP4** (task 172) | `NVFP4` | E2M1 | UE8M0 per block 32 | 2 codes/byte, even code low nibble; 4 exponents little-endian per int32 | `quantize_nvfp4_indexer` matches `test/test_packed_layout_v2.py::_ref_nvfp4` exactly |
| **AWQ int4** (task 161) | `AWQ_INT4` | uint4 | affine: fp16 scale + int4 zero per group | 8 codes/int32 in AWQ shuffled order `[0,4,1,5,2,6,3,7]` | `awq_dequantize` matches `_ref_awq` exactly |

`PackedLayout` now validates a **named storage rule** per storage kind, so an
invalid combination (wrong pack width, scale encoding, nibble order or container
width) is rejected instead of silently accepted:

```
e4m3:  pack_bits=8, scale=e8m0,  scale_dtype=uint8, nibble_order=low_first,   container_bits=8
e2m1:  pack_bits=4, scale=ue8m0, scale_dtype=int32, nibble_order=low_first,   container_bits=8
uint4: pack_bits=4, scale=affine,scale_dtype=fp16,  nibble_order=awq_shuffled,container_bits=32
```

New codecs: `e2m1_code/e2m1_value` (threshold ladder, sign only when the
magnitude is non-zero), `ue8m0_ceil_exp/ue8m0_scale/ue8m0_value`, nibble and
exponent-word packers, `quantize_nvfp4_indexer`/`dequantize_nvfp4_indexer`,
`awq_dequantize`.

## What this unlocked

| kernel | status | evidence |
|---|---|---|
| NVFP4 quantiser (task 172) | **promoted**: `examples/packed/fp4_indexer.fgpu` | compiles to frontend operators; its reference output matches the NVFP4 oracle byte-for-byte (data bytes + packed scale word) in `test/test_packed_dsl.py`, over random and boundary inputs |
| AWQ dequantiser (task 161) | **promoted**: `examples/packed/awq_dequant.fgpu` | uses the new `interleave` layout operator to build AWQ's in-word nibble order; `test/test_packed_dsl.py` checks every column against the statement (fp16 output rounding included) |

The NVFP4 program needed **no new operator**: `reshape` + `split` separate the
even/odd codes for pairwise byte packing, and `split` on the group axis plus
shift/or builds the four-exponent int32 word.  Two details came from the task
statement and are pinned in the program comments: `amax` is over `|x|` (a signed
`max_last` gave the wrong block scale), and the scale division is explicitly
rounded to fp32 because the reference backend computes in fp64 while the task is
fp32 — both were caught by the byte-exact comparison, not by a tolerance test.

## The AWQ promotion needed one operator

AWQ's reorder is a permutation *inside* a word, which reshape/split/concat
cannot express: those keep last-axis blocks contiguous.  The new **`interleave`**
layout operator (`out[..., 2i] = a[..., i]`) builds a per-word axis by
alternating two same-shaped tensors; a seven-call tree realizes AWQ's
`[0,4,1,5,2,6,3,7]` order.  It is host-lowered like its siblings (torch `stack`
+ reshape on the Triton host, flat index loops in the reference), and the
frontend refuses mismatched shapes, mismatched dtypes and a symbolic last axis.

`docs/EXTERN_LEDGER.md` now lists both 4-bit candidates as promoted.

## v3 additions (2026-10-27)

Three descriptor extensions, all backward compatible with v1/v2:

| addition | detail |
|---|---|
| **E5M2 twin** | `MXFP8_E5M2` (`storage=e5m2`, 8-bit, E8M0 block scales).  Codec `e5m2_byte/e5m2_value` is pinned against the software encoder; the one tie the two independent oracles resolve differently (61440: infinity vs saturate) is asserted explicitly in the test rather than hidden. |
| **scale modes** | `scale_mode` is now `block` (default), `row`, or `tensor`; `scale_shape` reflects it and 4-bit storages keep requiring `block` (their format defines it). |
| **nibble order** | `pack_nibbles(codes, order=...)` supports `low_first` (NVFP4) and `high_first`; the descriptor still pins which convention a layout uses. |

## Boundaries

- **No DSL program yet** for either v2 layout: nibble packing needs a pairwise
  lane combine that the current operator set does not have, so no promotion is
  claimed.  (`MXFP8` did get a DSL program in v1 and is device-smoked nightly.)
- **NVFP4 per-tensor global scale** (SGLang's full NVFP4 has an extra fp32
  alpha) is not modelled; task 172's statement does not include one.
- **AWQ group size** is derived from the shapes (`K // scales.shape[0]`), as the
  statement does; no `-1` sentinel is accepted.
- Rounding is defined by the statements: NVFP4 is exact integer/threshold logic;
  AWQ is a pure affine unpack.  Both are exact-compare, not tolerance-compare.
