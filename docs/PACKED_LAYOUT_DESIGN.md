# Packed layout / bitfield descriptor — design v1 (2026-10-27)

Status: **v1 implemented for MXFP8** (descriptor + codecs + DSL program, see
`docs/PACKED_LAYOUT_V1.md`) and **v2 implements the descriptors + exact
reference oracles for NVFP4 (E2M1/UE8M0) and AWQ int4**
(`docs/PACKED_LAYOUT_V2.md`).  A DSL expression of 4-bit packing is the
remaining gap; E5M2 and per-row scales remain design-only.  This is the B-route answer to the most
repeated language gap: every quantisation task ends up hand-writing the same
"how do logical values map to storage" logic, because FlowGPU can express the
arithmetic but not the layout.

Evidence for building it (from `docs/PACKED_LAYOUT_LIBRARY.md`): FlagOS batch11
161/163/172/174 (AWQ nibbles, E4M3 bytes, E2M1+UE8M0, MXFP8), CANN
DynamicMxQuant/Q4_0/WeightQuant, and the extern kernels that exist only because
of this gap.  That is >10 tasks across two competitions, so it passes the
"two independent tasks" promotion rule.

## 1. Surface: layout descriptors on operands, not operators

Keep the operator set unchanged.  A descriptor is an attribute of an **operand
type**, so `x * 2` works the same whether `x` is fp32 or E4M3 bytes:

```
layout  e4m3                          # storage: one byte per value, E4M3 code
layout  e4m3(scale=per_block(32))     # + one fp32 scale per 32 values
layout  fp4_e2m1(scale=per_block(32)) # two 4-bit codes per byte, UE8M0 byte scales
layout  awq_i4(order=shuffled)        # 4-bit codes, AWQ nibble shuffle
layout  mxfp8(scale=per_row)          # E8M0 exponent byte per 32 values
```

Grammar (initial):

```
layout      := name ( '(' setting (',' setting)* ')' )?
setting     := 'scale' '=' scale
             | 'order' '=' name
             | 'pack' '=' name
             | 'align' '=' int
scale       := 'per_tensor' | 'per_row' | 'per_block' '(' int ')' | 'none'
```

Descriptor fields that actually affect codegen:
storage dtype (u8/fp8/i8), bit width (4/8), codes per byte, packing order
(little-endian / shuffled), scale dtype (fp32 / E8M0 / bf16) and granularity
(per-tensor / per-row / per-block N), padding/alignment.

## 2. IR

`Value` gains an optional `PackedLayout` record (frozen dataclass: storage
dtype, bits, pack_order, scale_dtype, scale_granularity, alignment).  The
regular TensorType keeps the **logical** dtype/shape; the layout is how the
buffer is stored.  Rules:

- A value with a layout only enters arithmetic through an explicit
  `unpack(value)` / `pack(value, layout)` node; implicit coercion is a
  compile error (this is what prevents silent layout bugs).
- Fusion anchors treat pack/unpack as opaque boundaries for now (same as
  matmul), so fusion does not have to reason about bit layouts in round one.
- The reference lowering can always materialise the logical tensor, so the
  oracle tests keep working unchanged.

## 3. Lowering plan

Phase 1 (E4M3 / E5M2 per-tensor and per-block 32; covers FlagOS 163/166/174 and
CANN mxfp8):
1. frontend: descriptor parse + type check (storage size, block size divides
   shape, scale shape derivation);
2. reference: `pack`/`unpack` as pure Python over the logical tensor; gives the
   fp64 oracle for free;
3. triton: `unpack` = byte load + code→float table arithmetic; `pack` = scale
   computation + round/clamp + byte store.  Both are elementwise over the
   logical index space, so they reuse the existing pointwise emitter; the new
   part is address arithmetic (`logical_index → byte_index`) and the scale
   index.
4. tests: (dtype × block size × tail shape) bit-exact vs reference, plus the
   existing extern kernels re-expressed as DSL programs and compared component
   by component.

Phase 2 (AWQ 4-bit, E2M1 4-bit, UE8M0 scales): sub-byte packing needs
read-modify-write for odd packs; layout descriptor gains `pack=2x4bit` and the
emitter uses a byte-level RMW kernel.  FlagOS 161/172 are the acceptance tests.

Phase 3 (interleaved/paged layouts, e.g. store_sf_interleaved): only if a third
task needs it; a `order=interleaved(N)` setting maps the logical index through a
small permutation table that the frontend resolves.

## 4. Verification and migration

- Every phase ships with (a) bit-exact reference comparison on the CPU double
  harness, (b) a Triton interpreter smoke test, (c) one migrated extern kernel;
- the extern kernels in `competition/cann_bench/extern_20261026/` and
  `competition/season2/batch11/work/` are the acceptance corpus; a phase is
  "done" only when its target extern kernel is deleted;
- performance test: the migrated program must be within 10 % of the extern
  kernel on the same platform (the extern version is the baseline).

## 5. Open questions

- Where do layout attributes live syntactically: pipeline input/output
  declarations only, or also intermediate values?  Start with inputs/outputs
  (that is where the ABI is), allow intermediates later.
- Scale rounding semantics: the platform references differ (FN vs RNE, clamping
  order).  Descriptor should not hide this; a `round=` setting is needed.
- Layout equality/aliasing: two descriptors are equal only if every field
  matches; the memory planner must never reuse a buffer under a different
  layout.
- Do not let the reference lowering silently accept a layout mismatch: the
  oracle is only useful if it fails exactly where the platform would.
