"""Explicit pointwise mapping profiles; no device probing or performance claims.

Ascend's loop profile follows its vector-operator development guide, but has
not been executed on an Ascend device. Core counts must come from the caller's
actual Vector Core properties, not a guessed chip-family constant.
"""

from dataclasses import asdict, dataclass, replace
from types import MappingProxyType


MAPPING_STRATEGIES = ("direct_grid", "grid_stride_loop")
TRANSPOSE_MAPPINGS = ("tiled", "output_gather")
INDEX_DTYPES = ("auto", "int32", "int64")
#: How the value layer spells `sigmoid`. `native` is `tl.sigmoid(x)`; `exp2` is
#: the algebraically identical `1 / (1 + exp2(-log2(e) * x))`, whose only
#: transcendental is a base-2 exponential. The second form exists because on a
#: backend whose libdevice `exp` is emulated elementwise rather than mapped to a
#: native vector instruction, that one function can dominate a fused kernel.
SIGMOID_IMPLS = ("native", "exp2")
#: The largest address a signed 32-bit program index may derive. A kernel that
#: emits int32 index arithmetic is only correct while every element offset it
#: computes from that index stays inside this bound; the emitted host enforces
#: the bound from runtime shape/stride metadata before every launch.
MAX_INDEX32 = 2**31 - 1

#: Kernel kinds that have a hardware receipt on Ascend from the CANN-Bench runs.
#: The default `ascend` profile deliberately allows only `pointwise`; the named
#: `ascend-wide` profile exposes this wider set to tools/tests that opt in, and
#: `tools/cann_bench/gen_from_dsl.py` generated the evidence for it.
ASCEND_EVIDENCE_KINDS = (
    'pointwise', 'rowwise', 'matmul', 'transpose', 'rope', 'layout', 'primitive',
    'scan', 'topk', 'scatter_by_index', 'axis_gather', 'recurrence', 'cache_store',
    'store_at', 'pad_cat', 'linear_decode', 'patchify', 'segmented_matmul',
    'reduce_with_index', 'index_table',
)


@dataclass(frozen=True)
class PlatformConfig:
    name: str
    mapping_strategy: str = "direct_grid"
    default_block_size: int = 1024
    preferred_grid_multiplier: int = 1
    min_align_bytes: int = 1  # documented local-storage hint, not a pointer guarantee
    num_warps: int = 4
    num_stages: int = 2
    # Conv implicit-GEMM tile caps; runtime tuning can override these per
    # shape without changing any default source.
    conv_block_m: int = 128
    conv_block_n: int = 128
    conv_block_k: int = 64
    group_m: int = 1  # 1 = direct (pid_m, pid_n); >1 = L2 Cache Group Swizzling (e.g. 8)
    device_type: str = "cuda"
    device_cores: int | None = None
    # Largest grid extent the launch path accepts, independent of any tuning
    # choice. Ascend rejects coreDim > 65535 with Invalid_Argument (EE1003), as
    # recorded in the official Task155 receipt of 2026-10-18. None means no
    # measured ceiling, so the launch keeps the exact tile grid.
    max_grid_dim: int | None = None
    # Measured launch-parameter curve, as (value, relative-to-peak) pairs. This is
    # *evidence*, not policy: it records where a measured optimum sits and how
    # narrow it is, so a candidate search can consult it instead of guessing
    # powers of two. It deliberately does NOT change `default_block_size` --
    # every point below was measured on a single memory-bound operator, and a
    # platform-wide default inferred from one kernel is exactly the kind of
    # over-generalisation this project keeps having to retract.
    #
    # 昆仑芯, Task144 copy kernel, official harness 2026-10-20, block sizes
    # relative to the 2048 peak:
    #   512 -> 0.39   1024 -> 0.65   2048 -> 1.00   4096 -> 0.09
    # The peak is narrow: sampling only 512/1024/4096 concluded "0.15x is the
    # ceiling", and the untested 2048 turned out to be 53% better.
    measured_block_curve: tuple[tuple[int, float], ...] | None = None
    # --- Backend code-generation contracts (measured, not inferred) -----------
    # Whether a *masked load that passes `other=`* returns that filler value on
    # the lanes the mask excludes. 昆仑芯 does not: eleven official Task144
    # evaluations on 2026-10-19 isolated it to this single construct (attempts
    # 6 vs 7 differ only in the load mask and flip correctness), while *store*
    # masks are reliable there. On such a target, ownership of the zero border
    # has to come from an allocation, not from `other`.
    masked_load_other_honoured: bool = True
    # Whether `tl.make_block_ptr` with `boundary_check` / `padding_option`
    # produces correct values. Ascend does not: the same 2-D tiled kernel is
    # 55/55 exact on H100 and 2.7%-14.7% wrong on 华为. The working substitute is
    # a 2-D `tl.arange` tile with explicit masks.
    block_ptr_reliable: bool = True
    # Whether the target wants per-program *2-D* tiles with explicit strides
    # instead of a flat element range. On Ascend this is the single axis that
    # moved performance: 0.07x -> 0.17x on Task144 华为, while seven launch-level
    # knobs (num_warps, BLOCK, compile_mode, care_padding) were all neutral.
    # Consumed by `indexed_backend.triton_kernel`: its flat indexed form emits
    # one contiguous run of a FLAT index space, and every strided input view is
    # then addressed through `idx // dim1 * stride0 + idx % dim1 * stride1` --
    # a division/modulo the backend cannot prove contiguous. The 2-D form
    # addresses with `rows[:, None] * stride0 + cols[None, :]` derived from two
    # program ids, so the per-row run is a provable contiguous tile.
    vector_tile_2d: bool = False
    # The per-program tile shape the consumer above uses: `tile_rows` rows of
    # an output, each a contiguous `tile_cols`-element run of one row. Kept as
    # two fields (not one element count) because the measured axis is the
    # *shape*: the 华为 Task144 tile that went 0.07x -> 0.17x was a 2-D tile
    # whose contiguous axis was 512 elements. 1 x 1 is "no tile", which is the
    # default and keeps every platform that has not measured this axis on its
    # existing source.
    tile_rows: int = 1
    tile_cols: int = 1
    # Whether a *gather* load -- one whose per-lane address is decoded out of the
    # flat index with `//` or `%` -- is affordable. 昆仑芯 is not: the Task154
    # cross-chip `normalize_and_patchify` kernel is 1.81x on 国际通用 A and
    # **0.02x** there, a 50x collapse that drops under the 0.1x performance gate
    # and voids the whole submission. The same operator rewritten so that the
    # load address is the flat source index (a permutation moved into the store
    # address) is 0.05x -- better, still under the gate, so the *class* is
    # unusable there, not just the first form.
    gather_load_ok: bool = True
    # Whether a `tl.store` may sit *inside* a loop over a data axis, i.e. whether
    # a serial per-element dependence chain is affordable. 昆仑芯 is not: the
    # Task152 `kda_gate_chunk_cumsum` kernel walks a chunk of time with a running
    # accumulator and one store per step, and measures 0.03x there both with a
    # runtime-bounded loop and with a `tl.static_range` unroll -- so it is the
    # loop-carried store, not the loop form, that the backend cannot sustain.
    # Row-wise kernels that accumulate in registers and store once (Task147
    # rmsnorm family, 6.28x on that chip) are unaffected.
    store_inside_loop_ok: bool = True
    # Whether a *write* address must be `scalar_base + affine(arange)`. 昆仑芯
    # and 华为 are recorded as accepting only that shape: the hand-written
    # Task153 module deliberately keeps every store on `base + constexpr * row +
    # arange` and states "that is the shape 昆仑芯/华为 accept", while the two
    # programs that collapsed there (Task152 0.03x, Task154 0.02-0.05x) were the
    # ones whose addresses were not that shape. The check is enforced before a
    # kernel is released by `analysis/write_address.require_affine_writes`; it
    # reads the emitted source, names the address and the lane dependency, and
    # is a *classification* (affine / lane division-modulo / data-dependent),
    # not a device model. False (the default and every CUDA-class target) keeps
    # the emitted bytes exactly as they are.
    affine_write_addresses: bool = False
    tuning_status: str = "untuned"
    allowed_kernel_kinds: tuple[str, ...] = ("pointwise",)
    transpose_mapping: str = "tiled"
    # Longest row a single-program reduction may hold in registers at once.
    # A longer row is lowered as a chunked two-pass reduction rather than
    # compiled into a block the target cannot sustain. None means the target
    # has no measured reduction ceiling yet.
    reduction_block_limit: int | None = None
    # A candidate space is only published when compile-safe platform evidence
    # exists. None/unavailable must not silently inherit CUDA guesses.
    autotune_status: str = "unavailable"
    # Explicit experimental R/D tile bounds for indexed reductions. These
    # do not select a policy or inherit hardware guesses on other targets.
    indexed_columns: int | None = None
    indexed_rows: int | None = None
    gather_mapping: str = 'flat'
    # Explicit experiments; preserve existing placement and target defaults.
    reduction_group_limit: int = 2
    coalesce_reduction_sweeps: bool = False
    experimental_region_fusion: bool = False
    # Explicit low-level reduction ABI for a caller-owned strided workspace.
    # Generated hosts still allocate contiguous slots; no automatic padding.
    strided_reduction_outputs: bool = False
    # FlagTree-XPU vendor compile option. Never infer this on other targets.
    xpu_buffer_size_limit: int | None = None
    # Opt-in publication of visible row statistics when ready, before later
    # vector sweeps. This changes scheduling, not placement or arithmetic.
    eager_row_result_stores: bool = False
    # Explicit experiment: replace coordinate reconstruction by a linear
    # stride only when the runtime constexpr metadata proves it equivalent.
    linear_view_indexing: bool = False
    # Runtime metadata must prove adjacent interleaved external views before
    # one two-lane load may replace two independent reads. No guessed aliasing.
    paired_view_loads: bool = False
    # Module-level ABI contract for a pair of external inputs that are the two
    # contiguous halves of one row-major buffer (`gate = x[..., :N]`,
    # `up = x[..., N:]` of `x: [M, 2N]`). The *program* still declares two
    # same-shape inputs, so no arithmetic changes; the flag tells the lowering
    # that the declared last axis `N` is half of the caller's last axis and
    # that both views are addressed off ONE base pointer:
    #     gate address = idx + (idx // N) * N        up address = gate + N
    # This is the lowering the DSL's `split` cannot express at run time: split
    # requires a literal last axis and the language has no dimension arithmetic
    # for `2N`, so a typed `gate, up = split(gate_up)` materialises two host
    # copy kernels instead (measured 2026-10-21). The flag is therefore the
    # caller's contract, proven by construction in the adapter (one contiguous
    # `[M, 2N]` buffer passed for both inputs, exactly what `reshape(-1, 2N)`
    # produces), never guessed from tensor metadata.
    split_half_inputs: bool = False
    # Emit the per-call input-contract guards in a generated host. True is the
    # language's portability contract and stays the default. False is for a
    # module whose own entry point has already validated the call (the
    # competition adapter does) and where the guard block is measured host
    # time: Task160's host breakdown attributed 5.1 us of a 31.8 us call to it
    # (H100, guarded, 2026-10-21). Kernels, allocations and launch arguments
    # are unchanged by this flag -- only the guards in front of them.
    runtime_validation: bool = True
    # Emit the shared 32-bit address-range guard the flat emitters need (review
    # P1-8, 2026-10-26).  The default emitters address with int32 arithmetic
    # whatever the kernel metadata says, so an input that can span more than
    # 2**31 - 1 addresses would wrap silently; with this on, the host rejects
    # the call instead.  False is for a module whose shapes are known small and
    # whose bytes are a measured anchor (the Task160 package family): turning it
    # off there keeps the already-scoring modules byte-identical.
    index_guard: bool = True
    # Emit a mask-free kernel specialisation for the flat indexed pointwise
    # form when the launch covers every lane exactly. `numel % block == 0`
    # means `idx < numel` is true for every lane of every program, so the mask
    # is provably redundant and the kernel may read and store without it. The
    # kernel carries one extra `tl.constexpr` tail flag; the host passes
    # `(M*N) % BLOCK != 0`, and Triton compiles one binary per flag value, so
    # the exact-multiple case contains no predicates. That is the second,
    # mask-free kernel a measurement wants, without a second emitter or a
    # second ABI entry point. False keeps every already-scoring module
    # byte-identical (the emitted source does not change at all).
    mask_elision: bool = False
    # Host-lowered layout operators (reshape/split/concat/interleave) copy
    # through torch by default.  True emits a device copy kernel instead, so a
    # generated module contains no host tensor transform (a submission-legality
    # requirement for FlagOS/CANN and a speed requirement for pack/unpack
    # pipelines).  Default False keeps every existing module byte-identical.
    layout_on_device: bool = False
    # Whether this target has a *validated* layout-view fusion contract (P0):
    # a layout anchor (split/concat/reshape/interleave) fused into its pointwise
    # consumer as one indexed region.  CUDA is on because three real FlagOS
    # tasks were device-validated; non-CUDA targets stay off until their indexed
    # layout contracts are measured.  compile_source(None) consults this field;
    # explicitly asking for layout_views on a False target is refused rather
    # than emitting an unvalidated indexed region.
    layout_views: bool = False
    # Streaming eviction hint on the one-pass loads and store. Every input
    # element is read exactly once and every output element written once, so
    # `evict_first` is the textbook policy: it tells the cache hierarchy this
    # line has no reuse. It is a *hint* -- it cannot change values -- and the
    # spelling is the Triton 3.3.1 keyword `eviction_policy="evict_first"` on
    # `tl.load`/`tl.store`, the same spelling the vendored FlagGems library in
    # this tree uses (the vendored FlagGems softmax source),
    # which is the strongest local evidence that the FlagTree-based backends
    # accept it. Off by default; a backend that ignores the policy produces the
    # same binary as before.
    streaming_hints: bool = False
    # Experimental indexed pointwise mapping: keep [M,1] broadcasts scalar
    # while walking one bounded column tile from a single output row.
    broadcast_row_tiling: bool = False
    # Program-index width: 'auto' | 'int32' | 'int64'. int64 is the portable
    # choice and keeps the emitted source of every already-scoring target
    # unchanged. 昆仑芯 and 华为 pay a steep cost for 64-bit index arithmetic,
    # so their profiles ask for int32 -- but a lowering may emit int32 only
    # when it can also hand the host an enforceable address bound (see
    # `emitted_index_dtype`); an unbounded kernel keeps int64.
    index_dtype: str = "auto"
    # Sigmoid spelling: 'native' | 'exp2'. `native` is what every measured
    # target emits today and must stay the cuda default, because the emitted
    # source of the CUDA-class chips is part of their score. `exp2` asks for
    # the form whose exponential is a base-2 power, for a target whose exp is
    # emulated rather than native. It is the Task160 follow-up experiment:
    # with index width already falsified (int64 -> int32 moved 昆仑芯 1.57 ->
    # 1.59 and 华为 0.78 -> 0.79 while the five untouched chips reproduced
    # within ~1%), the sigmoid lowering is the next controlled variable.
    sigmoid_impl: str = "native"

    # --- GEMM tile policy (None = keep the historical 64x64x32 emitter) -----
    # The default lowering was a correctness-first 64x64x32 tile with no
    # software pipelining; the 2026-10-26 GPU MODE matmul_v2 run measured
    # 578.8 us there versus 241.6 us for a grouped 128x256x32 tile with a
    # 4-stage pipeline (711 TFLOP/s on H100).  These four fields let a policy
    # or a submission builder select the measured tile without editing the
    # emitter, and None preserves byte-identical output for every existing
    # platform.
    matmul_block_m: int | None = None
    matmul_block_n: int | None = None
    matmul_block_k: int | None = None
    matmul_num_warps: int | None = None
    matmul_num_stages: int | None = None

    def __post_init__(self):
        if type(self.reduction_group_limit) is not int or self.reduction_group_limit <= 0:
            raise ValueError('reduction_group_limit must be a positive integer')
        for name in ('coalesce_reduction_sweeps', 'experimental_region_fusion', 'strided_reduction_outputs', 'eager_row_result_stores', 'linear_view_indexing', 'paired_view_loads', 'broadcast_row_tiling', 'split_half_inputs', 'runtime_validation', 'index_guard',
                     'mask_elision', 'streaming_hints', 'layout_on_device', 'layout_views',
                     'mask_elision', 'streaming_hints'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be boolean')
        if self.xpu_buffer_size_limit is not None:
            if (not self.name.startswith('kunlun') or type(self.xpu_buffer_size_limit) is not int
                    or self.xpu_buffer_size_limit<=0):
                raise ValueError('xpu_buffer_size_limit requires an explicit Kunlun target and positive integer')
        if self.gather_mapping not in {'flat', 'row_tiled'}:
            raise ValueError('gather_mapping must be flat or row_tiled')
        for name in ("indexed_columns", "indexed_rows"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0 or value & (value - 1)):
                raise ValueError(f"{name} must be a positive power of two")
        if self.indexed_columns is not None and self.indexed_rows is not None:
            if self.indexed_columns * self.indexed_rows > (self.reduction_block_limit or 4096):
                raise ValueError("indexed R/D tile exceeds reduction_block_limit")
        if self.mapping_strategy not in MAPPING_STRATEGIES:
            raise ValueError(f"unknown mapping_strategy: {self.mapping_strategy}")
        if self.broadcast_row_tiling and (self.name != 'cuda' or self.mapping_strategy != 'direct_grid'
                                         or self.max_grid_dim is not None):
            raise ValueError('broadcast_row_tiling currently requires CUDA direct_grid without a grid cap')
        if self.transpose_mapping not in TRANSPOSE_MAPPINGS:
            raise ValueError(f"unknown transpose_mapping: {self.transpose_mapping}")
        if self.index_dtype not in INDEX_DTYPES:
            raise ValueError("index_dtype must be auto, int32 or int64")
        if self.sigmoid_impl not in SIGMOID_IMPLS:
            raise ValueError("sigmoid_impl must be native or exp2")
        for name in ("default_block_size", "preferred_grid_multiplier", "min_align_bytes", "num_warps", "num_stages", "group_m"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("default_block_size", "min_align_bytes", "num_warps"):
            value = getattr(self, name)
            if value & (value - 1):
                raise ValueError(f"{name} must be a power of two")
        for name in ("tile_rows", "tile_cols"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value & (value - 1):
                raise ValueError(f"{name} must be a power of two")
        for name in ("matmul_block_m", "matmul_block_n", "matmul_block_k",
                     "matmul_num_warps", "matmul_num_stages"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.device_cores is not None and (type(self.device_cores) is not int or self.device_cores <= 0):
            raise ValueError("device_cores must be a positive integer")
        if self.max_grid_dim is not None and (type(self.max_grid_dim) is not int or self.max_grid_dim <= 0):
            raise ValueError("max_grid_dim must be a positive integer")
        if self.reduction_block_limit is not None and (
                type(self.reduction_block_limit) is not int
                or self.reduction_block_limit <= 0
                or self.reduction_block_limit & (self.reduction_block_limit - 1)):
            raise ValueError("reduction_block_limit must be a positive power of two")
        if self.mapping_strategy == "grid_stride_loop" and self.device_cores is None:
            raise ValueError("grid_stride_loop requires explicit device_cores (Ascend: Vector Core count)")
        if self.device_type not in {"cuda", "npu"}:
            raise ValueError("device_type must be 'cuda' or 'npu'")
        if self.measured_block_curve is not None:
            curve = self.measured_block_curve
            if not isinstance(curve, tuple) or len(curve) < 2:
                raise ValueError("measured_block_curve must be a tuple of at least two points")
            for point in curve:
                if (not isinstance(point, tuple) or len(point) != 2
                        or type(point[0]) is not int or point[0] <= 0
                        or point[0] & (point[0] - 1)
                        or type(point[1]) is not float or not 0.0 < point[1] <= 1.0):
                    raise ValueError(f"bad measured_block_curve point: {point!r}")
            if sorted(v for v, _ in curve) != [v for v, _ in curve]:
                raise ValueError("measured_block_curve must be sorted by block size")
            if sum(1 for _, r in curve if r == 1.0) != 1:
                raise ValueError("measured_block_curve must have exactly one peak (relative 1.0)")
        for name in ("masked_load_other_honoured", "block_ptr_reliable", "vector_tile_2d",
                     "gather_load_ok", "store_inside_loop_ok", "affine_write_addresses"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        if self.autotune_status not in {"available", "experimental", "unavailable"}:
            raise ValueError("autotune_status must be available, experimental or unavailable")
        if (not isinstance(self.allowed_kernel_kinds, tuple)
                or not self.allowed_kernel_kinds
                or any(kind not in {"pointwise", "rowwise", "matmul", "transpose", "rope", "layout", "primitive", "cache_store", "linear_decode", "pad_cat", "scan", "patchify", "segmented_matmul", "scatter_by_index", "store_at", "topk", "sort", "mean", "layer_norm", "conv2d", "conv2d_same", "conv2d_stride2", "conv2d_stride2_same", "axis_gather", "reduce_with_index", "index_table", "recurrence"}
                       for kind in self.allowed_kernel_kinds)):
            raise ValueError("allowed_kernel_kinds must be a nonempty tuple of known kernel kinds")

    @property
    def grid_limit(self):
        if self.mapping_strategy == "direct_grid":
            return None
        return self.device_cores * self.preferred_grid_multiplier

    def as_dict(self):
        return asdict(self)


# These are compatibility/search starting points, not measured optimums.
# Ascend's strategy is selected in resolve_platform only after a core count
# is supplied. The profile table itself remains a valid immutable object.
#
# The two measured reduction_block_limit entries come from opposite ends:
# - kunlunxin 128: the batch-8 dcp_lse_combine submissions showed a
#   next_power_of_2(D)-wide reduction tile dying in TritonXPU's MLIR pipeline
#   with `OutOfResources: uni_sram`, while 128-lane kernels compiled and ran.
# - cuda 4096: A800 CUDA-graph timing (internal evidence,
#   45 configs, 0 wrong, sanity pair 0.0%) shows the single-tile form winning
#   up to D=16384 (-27%..-37% vs chunked) and collapsing ~10x from D=32768
#   (2088us vs 233us at width 16384); chunk width is flat 244-254us across
#   4096-16384, so 4096 is the safe measured default.
# A longer or symbolic row is lowered as a chunked two-sweep reduction
# instead of one oversized tile.
_BASE_PLATFORMS = {
    "cuda": PlatformConfig("cuda", allowed_kernel_kinds=("pointwise", "rowwise", "matmul", "transpose", "rope", "layout", "primitive", "cache_store", "linear_decode", "pad_cat", "scan", "patchify", "segmented_matmul", "scatter_by_index", "store_at", "topk", "sort", "mean", "layer_norm", "conv2d", "conv2d_same", "conv2d_stride2", "conv2d_stride2_same", "axis_gather", "reduce_with_index", "index_table", "recurrence"),
                           # Default GEMM floor, measured 2026-10-27 on GPU MODE H100:
                           # 128x128x64 / 3 stages / 8 warps runs the ranked
                           # 4096x4096x5120 shape in 286 us versus 578.8 us for the
                           # historical 64x64x32 tile, and its 96 KB of shared
                           # memory fits every CUDA-class device measured so far
                           # (L4 caps at 101 KB).  The grouped 128x256x64 policy
                           # (227.4 us H100) remains opt-in per platform because
                           # its swizzled grid changes emission structure.
                           matmul_block_m=128, matmul_block_n=128, matmul_block_k=64,
                           matmul_num_warps=8, matmul_num_stages=3,
                           reduction_block_limit=4096, autotune_status="available",
                           # Keep the emitted source of every CUDA-class chip
                           # byte-identical: int64 index arithmetic is what
                           # they already score 4-11x with.
                           index_dtype="int64",
                           # ...and so does the native sigmoid spelling. This
                           # pin is the byte-identity contract for the five
                           # already-scoring chips, not a measured optimum.
                           sigmoid_impl="native", layout_views=True),
    # Ascend: masked loads honour `other` (every 华为 kernel that used them stayed
    # numerically correct), `make_block_ptr` does not (Task144 attempt 15), and
    # per-program 2-D tiles are the axis that moves performance (attempt 16).
    # The sigmoid spelling is native, and that is a measured decision: the exp2 form
    # was submitted to 华为 on 2026-10-21 (attempts 9 and 10) and read 0.75x and
    # 0.76x against 0.72-0.79x before it, i.e. inside the noise. The transcendental
    # lowering is not what makes a fused kernel lose to eight eager ops on this
    # backend, so the proven `tl.sigmoid` stays. What the platform record does say
    # moves performance here is the *tile shape* (`vector_tile_2d`), so the
    # indexed emitter now reads that field and the shape below.
    #
    # 2 x 512, not 4 x 256: the product is 1024, i.e. exactly the element count
    # the flat kernel gives each program, so program count and per-program work
    # are unchanged and the tile *shape* is the only variable (the axis the
    # platform record names). Between the two shapes that budget allows, 512 is
    # the contiguous run length -- 1 KiB per row burst in fp16 -- and the
    # Task144 华为 tile that measured 0.07x -> 0.17x had 512 elements on its
    # contiguous axis with more rows on the strided one; 4 x 256 would halve the
    # contiguous run and triple the row discontinuities per program. It also
    # bounds the lane waste on the eval sweep's many M=1 shapes at 2x instead of
    # 4x, where those shapes are host-bound anyway.
    "ascend": PlatformConfig("ascend", device_type="npu", min_align_bytes=32, allowed_kernel_kinds=("pointwise",),
                             max_grid_dim=65535, block_ptr_reliable=False, vector_tile_2d=True,
                             tile_rows=2, tile_cols=512,
                             index_dtype="int32",
                             sigmoid_impl="native",
                             affine_write_addresses=True,
                             # Task144 华为, 2026-10-20: num_warps 1/4/16 and BLOCK
                             # 4096/8192/16384 all reproduced the same reading, so
                             # the launch parameters are flat here and the tile
                             # *shape* is the axis that matters. A flat curve with
                             # no single peak is not representable, so no curve is
                             # recorded: that absence is the finding.
                             ),
    "ascend-910b": PlatformConfig("ascend-910b", device_type="npu", min_align_bytes=32, max_grid_dim=65535,
                                  block_ptr_reliable=False, vector_tile_2d=True,
                                  tile_rows=2, tile_cols=512, index_dtype="int32",
                                  sigmoid_impl="native", affine_write_addresses=True,
                                  # CANN softmax case 17 (row 8193) aborted with an
                                  # empty device error while every row <= 4096 passed;
                                  # the one-tile form would put a 16384-lane vector on
                                  # the vector core. 4096 makes the rowwise lowering
                                  # take its measured two-sweep chunked form, exactly
                                  # as the CUDA profile does for the same reason.
                                  reduction_block_limit=4096),
    # 昆仑芯: a masked load does not honour `other`; store masks do. The 2-D tile
    # that fixed 华为 does not port here even with a zero-owned border
    # (Task144 attempt 18), so `vector_tile_2d` stays False.
    # The gather-load and loop-carried-store bans come from six official
    # submissions on 2026-10-20 (Task146/147/149 accepted at 0.60x / 6.28x /
    # 2.01x; Task152/154 rejected at 0.03x / 0.02-0.05x). What survives there is
    # row-wise work: coalesced load, register accumulation, one affine store.
    "kunlunxin": PlatformConfig("kunlunxin", allowed_kernel_kinds=("pointwise",), transpose_mapping="output_gather",
                               reduction_block_limit=128, autotune_status="experimental", index_dtype="int32",
                               # Native sigmoid, deliberately. This repo's own
                               # batch-8 evidence says `tl.exp2` is the primitive
                               # that no passing 昆仑芯 submission had used and
                               # that was a named suspect in the LSE numerical
                               # failures, whose fix replaced exp2 with
                               # exp(x*ln2) -- see
                               # docs/EVALUATION_MAPPING_FOLLOWUP.md 5-6. The
                               # proven form here is `tl.sigmoid` = 1/(1+exp).
                               # So the Task160 transcendental experiment runs on
                               # 华为 ONLY, and 昆仑芯 stays a control: its module
                               # is byte-identical to the one that scored 1.59x.
                               # A numerical 0/7 on an unproven primitive is not
                               # worth one chip's measurement.
                               sigmoid_impl="native",
                               masked_load_other_honoured=False,
                               gather_load_ok=False, store_inside_loop_ok=False,
                               affine_write_addresses=True,
                               measured_block_curve=((512, 0.39), (1024, 0.65), (2048, 1.0), (4096, 0.09))),
    "kunlun-r300": PlatformConfig("kunlun-r300", transpose_mapping="output_gather", index_dtype="int32",
                                  sigmoid_impl="native"),
}

#: Named profiles are *views* of a measured base profile, never new defaults:
#: `cuda-tuned` switches on the grouped GEMM policy that measured 227.4 us on
#: H100 (the CUDA default stays the device-safe 128x128x64 floor), and
#: `ascend-wide` exposes the CANN-evidence kinds while keeping every measured
#: Ascend field (tile shape, grid cap, sigmoid spelling) unchanged.
PLATFORMS = MappingProxyType({
    **_BASE_PLATFORMS,
    "cuda-tuned": replace(_BASE_PLATFORMS["cuda"], name="cuda-tuned",
                          matmul_block_n=256, matmul_num_stages=4, group_m=8),
    "ascend-wide": replace(_BASE_PLATFORMS["ascend"], name="ascend-wide",
                           allowed_kernel_kinds=ASCEND_EVIDENCE_KINDS),
})

#: Profiles that admit a wider set of kernel kinds than the conservative default.
WIDER_PROFILES = {"ascend": "ascend-wide"}


def wider_profile_hint(config, kind) -> str:
    """A refusal hint that names the wider profile, when one admits the kind."""
    wider = WIDER_PROFILES.get(config.name)
    if wider is None or kind not in PLATFORMS[wider].allowed_kernel_kinds:
        return ''
    return ("; profile %r admits this kind (the CANN generator's evidence, see "
            "docs/BACKEND_COVERAGE_20261027.md)" % wider)


def measured_block_peak(curve):
    """The block size whose measured relative value is 1.0 -- the curve's peak.

    This is the single place the peak rule lives. Two consumers read it:
    :func:`block_candidates` takes the peak plus its neighbours as the search
    window, and :func:`resolve_platform` makes the peak the *default* block
    when the caller does not pass ``block_size``. If either derived the peak
    itself the search window and the default could silently disagree about
    where the measurement says the optimum is.

    ``None`` when there is no measured curve: absent measurement is not
    evidence of a flat surface, so a platform without one keeps its compiled
    default rather than inheriting a guess. ``PlatformConfig.__post_init__``
    already rejects a malformed curve (fewer than two points, unsorted sizes,
    or not exactly one relative-1.0 peak); the defensive checks here keep the
    helper total for synthetic curves a test may pass directly.
    """
    if not curve:
        return None
    peaks = [size for size, relative in curve if relative == 1.0]
    if len(peaks) != 1:
        raise ValueError(
            'measured_block_curve must have exactly one peak (relative 1.0), '
            f'got {len(peaks)} in {curve!r}')
    return peaks[0]


def resolved_default_block(platform):
    """The block a caller gets when it does not pass ``block_size``.

    The measured peak when the platform carries a measured curve, otherwise the
    platform's stored ``default_block_size``. One function so the default and
    :func:`block_candidates` read the same evidence through the same rule.
    """
    peak = measured_block_peak(platform.measured_block_curve)
    return peak if peak is not None else platform.default_block_size


def block_candidates(platform, *, floor=256, ceiling=8192):
    """Block sizes a search should actually try on this platform.

    Derived from ``measured_block_curve``: the measured peak plus its immediate
    neighbours. The point is to make a *narrow* optimum reachable. On 昆仑芯 a
    power-of-four sweep (512 / 1024 / 4096) concluded the ceiling was 0.15x,
    while the untested 2048 was 0.23x -- 53% better than the best sampled point
    (Task144, official harness, 2026-10-20). Asking the platform for the peak and
    its neighbours is how a search avoids repeating that.

    Returns an empty tuple when the platform has no measured curve: absent
    measurement is not evidence of a flat surface, so the caller's own candidate
    list applies rather than a guess derived from nothing.
    """
    curve = platform.measured_block_curve
    peak = measured_block_peak(curve)
    if peak is None:
        return ()
    sizes = [value for value, _ in curve]
    at = sizes.index(peak)
    chosen = [peak]
    if at > 0:
        chosen.insert(0, sizes[at - 1])
    if at + 1 < len(sizes):
        chosen.append(sizes[at + 1])
    return tuple(sorted(v for v in chosen if floor <= v <= ceiling))


def resolve_platform(name="cuda", *, mapping_strategy=None, block_size=None,
                     device_cores=None, grid_multiplier=None, num_warps=None,
                     num_stages=None, group_m=None, allowed_kernel_kinds=None,
                     transpose_mapping=None, reduction_block_limit=None,
                     autotune_status=None, indexed_columns=None, indexed_rows=None,
                     gather_mapping=None, reduction_group_limit=None,
                     coalesce_reduction_sweeps=None, experimental_region_fusion=None,
                     strided_reduction_outputs=None, xpu_buffer_size_limit=None,
                     eager_row_result_stores=None, max_grid_dim=None, linear_view_indexing=None,
                     paired_view_loads=None, broadcast_row_tiling=None,
                     masked_load_other_honoured=None, block_ptr_reliable=None,
                     vector_tile_2d=None, tile_rows=None, tile_cols=None,
                     gather_load_ok=None, store_inside_loop_ok=None,
                     affine_write_addresses=None,
                     index_dtype=None, sigmoid_impl=None,
                     split_half_inputs=None, runtime_validation=None,
                     layout_on_device=None,
                     index_guard=None,
                     mask_elision=None, streaming_hints=None,
                     matmul_block_m=None, matmul_block_n=None,
                     matmul_block_k=None, matmul_num_warps=None,
                     matmul_num_stages=None):
    if name not in PLATFORMS:
        raise ValueError(f"unknown platform: {name}; choose from {', '.join(PLATFORMS)}")
    is_ascend = name.startswith("ascend")
    changes = {"mapping_strategy": mapping_strategy if mapping_strategy is not None else (
        "grid_stride_loop" if is_ascend else "direct_grid")}
    for key, value in (("default_block_size", block_size), ("device_cores", device_cores),
                       ("preferred_grid_multiplier", grid_multiplier), ("num_warps", num_warps),
                       ("num_stages", num_stages), ("group_m", group_m),
                       ("transpose_mapping", transpose_mapping),
                       ("reduction_block_limit", reduction_block_limit),
                       ("allowed_kernel_kinds", allowed_kernel_kinds),
                       ("autotune_status", autotune_status)):
        if value is not None:
            changes[key] = value
    for key, value in (("matmul_block_m", matmul_block_m),
                       ("matmul_block_n", matmul_block_n),
                       ("matmul_block_k", matmul_block_k),
                       ("matmul_num_warps", matmul_num_warps),
                       ("matmul_num_stages", matmul_num_stages)):
        if value is not None:
            changes[key] = value
    # An explicit platform-wide launch knob is the operator/scheduler override:
    # it must beat the default GEMM floor, otherwise `--num-warps` silently stops
    # affecting GEMM on the profile that now pins a measured tile.  A matmul
    # *specific* override still takes precedence over the platform-wide one.
    if num_warps is not None and matmul_num_warps is None:
        changes['matmul_num_warps'] = None
    if num_stages is not None and matmul_num_stages is None:
        changes['matmul_num_stages'] = None
    if indexed_columns is not None: changes['indexed_columns'] = indexed_columns
    if indexed_rows is not None: changes['indexed_rows'] = indexed_rows
    if gather_mapping is not None: changes['gather_mapping'] = gather_mapping
    for key, value in (('reduction_group_limit', reduction_group_limit),
                       ('coalesce_reduction_sweeps', coalesce_reduction_sweeps),
                       ('experimental_region_fusion', experimental_region_fusion)):
        if value is not None: changes[key] = value
    if strided_reduction_outputs is not None: changes['strided_reduction_outputs']=strided_reduction_outputs
    if xpu_buffer_size_limit is not None: changes['xpu_buffer_size_limit']=xpu_buffer_size_limit
    if eager_row_result_stores is not None: changes['eager_row_result_stores']=eager_row_result_stores
    if max_grid_dim is not None: changes['max_grid_dim']=max_grid_dim
    if masked_load_other_honoured is not None: changes['masked_load_other_honoured']=masked_load_other_honoured
    if block_ptr_reliable is not None: changes['block_ptr_reliable']=block_ptr_reliable
    if vector_tile_2d is not None: changes['vector_tile_2d']=vector_tile_2d
    if tile_rows is not None: changes['tile_rows']=tile_rows
    if tile_cols is not None: changes['tile_cols']=tile_cols
    if gather_load_ok is not None: changes['gather_load_ok']=gather_load_ok
    if store_inside_loop_ok is not None: changes['store_inside_loop_ok']=store_inside_loop_ok
    if affine_write_addresses is not None: changes['affine_write_addresses']=affine_write_addresses
    if linear_view_indexing is not None: changes['linear_view_indexing']=linear_view_indexing
    if paired_view_loads is not None: changes['paired_view_loads']=paired_view_loads
    if broadcast_row_tiling is not None: changes['broadcast_row_tiling']=broadcast_row_tiling
    if index_dtype is not None: changes['index_dtype']=index_dtype
    if sigmoid_impl is not None: changes['sigmoid_impl']=sigmoid_impl
    if split_half_inputs is not None: changes['split_half_inputs']=split_half_inputs
    if runtime_validation is not None: changes['runtime_validation']=runtime_validation
    if index_guard is not None: changes['index_guard']=index_guard
    if mask_elision is not None: changes['mask_elision']=mask_elision
    if streaming_hints is not None: changes['streaming_hints']=streaming_hints
    if layout_on_device is not None: changes['layout_on_device']=layout_on_device
    if "default_block_size" not in changes:
        # ...and a measured block curve makes its PEAK the default, so a platform's
        # own measurement decides code generation instead of sitting in a table a
        # caller has to read. 昆仑芯's curve ((512, 0.39), (1024, 0.65), (2048, 1.0),
        # (4096, 0.09)) is official-board evidence, not a hardware guess: Task160 on
        # 昆仑芯 read 1.58x at the flat default 1024 and 1.78x / 1.85x / 1.86x at the
        # peak 2048 across submissions on 2026-10-21
        # (the two measured candidate variants/
        # submission-attempt-{13,15}.json). Before this, the only way a builder could
        # get the measured peak was to hard-code 2048, so every other caller silently
        # emitted the 0.65 relative point. cuda has no curve and does not move: a 2048
        # block there made 天数 worse (11.09x -> 8.36x), so "no measurement" keeps the
        # compiled default byte for byte.
        stored = PLATFORMS[name]
        resolved = resolved_default_block(stored)
        if resolved != stored.default_block_size:
            changes['default_block_size'] = resolved
    return replace(PLATFORMS[name], **changes)


def emitted_index_dtype(config, *, proven):
    """The program-index dtype a lowering may emit: ``'int32'`` or ``'int64'``.

    ``'int64'`` always wins, which is what keeps the emitted source of every
    already-scoring CUDA-class target unchanged. ``'int32'`` (a measured
    platform preference, 昆仑芯/华为) and ``'auto'`` emit 32-bit index
    arithmetic only when the emitter passes ``proven=True``: it has built an
    address bound the emitted host enforces against ``MAX_INDEX32`` before
    every launch. A kernel whose address extent cannot be bounded from the
    metadata the compiler has keeps int64 rather than run unchecked -- int32
    is never emitted with an unproven bound.
    """
    if config.index_dtype == "int64" or not proven:
        return "int64"
    return "int32"
