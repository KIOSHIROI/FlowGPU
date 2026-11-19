# Device-side layout lowering (T8, 2026-10-27)

## What changed

`reshape`, `split`, `concat` and `interleave` used to be **host-lowered** by both
backends: the generated module called `torch.reshape`/slicing/`torch.cat`/
`torch.stack` on tensor data. That made a DSL program containing them unsuitable
for submission (host-side core compute) and added host copies to pack/unpack
pipelines.

`PlatformConfig.layout_on_device = True` (CLI `--layout-on-device`) now emits a
**device copy kernel** per layout group instead:

| operator | device index mapping (flat output offset `o`) |
|---|---|
| `reshape` | identity copy (`src = o`) |
| `split` | `row = o // W; col = o % W; src = row*IN + part*W + col` |
| `concat` | `row = o // (A+B); col = o % (A+B)`; select `a[row*A+col]` or `b[row*B+col-A]` |
| `interleave` | `row = o // 2W; col = o % 2W`; select by parity from `a/b[row*W + col//2]` |

The last-axis widths are literals because the frontend already refuses symbolic
last axes for split/concat/interleave; `reshape` needs no width. Grids go through
`grid_limited_body`, so Ascend's `max_grid_dim` cap becomes a grid-stride loop
instead of an invalid launch. The buffer planner is unchanged (the planned slot
is filled by the kernel).

## Guarantees

- **Default off**: with the flag absent, emitted bytes and host lowering are
  unchanged (the byte-identity tests still pass).
- **Device correctness**: `test/test_layout_device.py` executes each of the four
  emitted kernels through the CPU Triton double and compares against the
  reference backend; a mixed concat→reshape→split→interleave program is also
  checked structurally for zero host transforms.
- **Submission legality**: with the flag on, the NVFP4 example's module contains
  no `torch.cat`, `torch.stack`, `copy_(` or slicing on tensor data.

## Boundaries

- This is a *copy* kernel per layout group, not view folding: it is correctness/
  legality first. The performance follow-up is to fold layout addressing into
  the consumer kernel (zero extra traffic).
- The device path is only emitted for profiles whose `allowed_kernel_kinds`
  include `layout` (cuda, ascend-wide, ...); a conservative profile still refuses.
- Reference backend keeps its host lowering as the numeric oracle; only the
  Triton module is submitted, so legality depends on the Triton side alone.
