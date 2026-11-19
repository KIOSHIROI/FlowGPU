"""Mapping configuration boundaries and emitted loop ownership."""

import ast
import json
from dataclasses import FrozenInstanceError
from math import isqrt

import pytest

from flowgpu.compiler.cli import main
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.platform_configs import PlatformConfig, resolve_platform
from flowgpu.compiler.triton_backend import module


POINTWISE = """pipeline point {
    input x : int32[N]
    input y : int32[N]
    t = add(x,y)
    z = mul(t,y)
    output z
}"""


def program(config=None, source=POINTWISE):
    return compile_source(source, platform_config=config).pipelines[0].triton


def test_default_profile_preserves_existing_direct_mapping():
    p = program()
    assert p.platform_config == resolve_platform("cuda")
    assert p.kernels[0].grid == "(triton.cdiv(N, BLOCK),)"
    assert "for " not in p.kernels[0].source
    assert p.kernels[0].launch_options == ()


def test_profiles_are_immutable_and_untuned():
    p = resolve_platform("kunlunxin")
    assert p.tuning_status == "untuned"
    assert p.device_cores is None
    with pytest.raises(FrozenInstanceError, match="cannot assign to field 'default_block_size'"):
        p.default_block_size = 4096


@pytest.mark.parametrize("kwargs", [
    {"block_size": 0}, {"block_size": 300}, {"block_size": True},
    {"device_cores": 0}, {"device_cores": -1}, {"device_cores": True},
    {"grid_multiplier": 0}, {"grid_multiplier": -1},
    {"num_warps": 3}, {"num_warps": 0}, {"mapping_strategy": ""},
    {"mapping_strategy": "automatic"}, {"mapping_strategy": "grid_stride_loop"},
])
def test_invalid_mapping_is_rejected_before_emission(kwargs):
    with pytest.raises(ValueError, match='(?:default_block_size must be a positive integer|default_block_size must be a power of two|device_cores must be a positive integer|preferred_grid_multiplier must be a positive integer|num_warps must be a power of two|num_warps must be a positive integer|unknown mapping_strategy:|unknown mapping_strategy: automatic|grid_stride_loop requires explicit device_cores \\(Ascend: Vector Core count\\))'):
        resolve_platform(**kwargs)


def test_ascend_requires_actual_vector_core_count():
    with pytest.raises(ValueError, match="device_cores"):
        resolve_platform("ascend")
    assert resolve_platform("ascend", device_cores=48).grid_limit == 48
    assert resolve_platform("ascend", mapping_strategy="direct_grid").grid_limit is None


def test_loop_wraps_entire_fused_expression_and_store():
    p = program(resolve_platform("cuda", mapping_strategy="grid_stride_loop", device_cores=3))
    k = p.kernels[0]
    tree = ast.parse(k.source)
    body = tree.body[0].body
    assert len(body) == 1 and isinstance(body[0], ast.For)
    assert k.ops == ("add", "mul")
    assert "tl.num_programs(0)" in k.source
    assert "no buffer" in k.source
    assert isinstance(body[0].body[-1], ast.Expr)
    assert body[0].body[-1].value.func.attr == "store"
    assert "else 3" in k.grid
    ast.parse(module([p]))


def test_per_program_options_do_not_leak_between_modules():
    small = program(resolve_platform(block_size=256, num_warps=8))
    large = program(resolve_platform(mapping_strategy="grid_stride_loop", device_cores=7,
                                      grid_multiplier=2, block_size=4096))
    # The configured block size is carried as a literal launch argument; the
    # layout metadata arguments (contiguous/shape/stride) follow the tensor
    # and dimension arguments, so locate the block literal by kind instead of
    # relying on launch[-1].
    def block_literal(p):
        return next(a.text for a in p.kernels[0].launch
                    if a.kind == "value" and (a.text.isdigit() or a.text == "BLOCK"))

    assert block_literal(small) == "256"
    assert "num_warps=8" in small.host
    assert block_literal(large) == "4096"
    assert "else 14" in large.kernels[0].grid
    assert block_literal(program()) == "BLOCK"


def test_ascend_host_checks_npu_and_imports_runtime():
    p = program(resolve_platform("ascend", device_cores=48))
    assert p.complete
    assert ".device.type != 'npu'" in p.host
    assert "same NPU device" in p.host
    assert "import torch_npu" in module([p])
    assert "torch.mm" not in p.host


def test_non_pointwise_foreign_lowering_is_explicitly_incomplete():
    source = "pipeline gemm { input x : fp32[M,K] input w : fp32[K,N] y = matmul(x,w) output y }"
    p = program(resolve_platform("ascend", device_cores=48), source)
    assert not p.complete
    assert not p.kernels
    assert "pointwise kernels only" in p.refusals[0].reason


def test_hybrid_is_rejected_for_foreign_platform():
    with pytest.raises(ValueError, match="gpu_backend"):
        compile_source(POINTWISE, gpu_backend="hybrid", platform_config=resolve_platform("kunlunxin"))


def test_new_loop_identifiers_do_not_shadow_tensor_or_dimension_names():
    source = "pipeline names { input min : int32[tile] input y : int32[tile] z = add(min,y) output z }"
    p = program(resolve_platform(mapping_strategy="grid_stride_loop", device_cores=3), source)
    assert "for tile_2 in " in p.kernels[0].source
    assert "min(" not in p.kernels[0].grid
    ast.parse(module([p]))


def test_cli_exposes_config_and_launch_metadata(tmp_path, capsys):
    src = tmp_path / "point.fgpu"
    src.write_text(POINTWISE)
    assert main([str(src), "--emit", "json", "--platform", "ascend", "--device-cores", "48",
                 "--block-size", "4096", "--num-warps", "8"]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    target = json.loads(captured.out)["pipelines"][0]["triton"]
    assert target["platform_config"]["device_cores"] == 48
    assert target["platform_config"]["tuning_status"] == "untuned"
    assert target["kernels"][0]["launch_options"] == {"num_warps": 8}


@pytest.mark.parametrize("flags", [
    ["--platform", "ascend"], ["--block-size", "300"],
    ["--mapping-strategy", "grid_stride_loop"],
    ["--platform", "kunlunxin", "--gpu-backend", "hybrid"],
])
def test_cli_invalid_config_is_usage_error(tmp_path, capsys, flags):
    src = tmp_path / "point.fgpu"
    src.write_text(POINTWISE)
    assert main([str(src), "--emit", "json", *flags]) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)["kind"] == "configuration"


def test_unknown_platform_and_wrong_api_object_are_rejected():
    with pytest.raises(ValueError, match="unknown platform"):
        resolve_platform("imaginary")
    with pytest.raises(TypeError, match="PlatformConfig"):
        compile_source(POINTWISE, platform_config={})


def test_ascend_caps_the_grid_to_the_hardware_extent():
    """Ascend rejects coreDim > 65535.

    The official Task155 receipt of 2026-10-18 failed on Ascend with
    ``Invalid_Argument(EE1003): value 65536 for parameter coreDim is invalid.
    Expected value: less than or equal to 65535``. A direct-grid pointwise
    launch of a 16.7M-element tensor at BLOCK=256 asks for exactly 65536
    programs, so the compiler must cap the grid and walk the rest in-kernel.
    """
    p = program(resolve_platform("ascend", mapping_strategy="direct_grid", block_size=256))
    k = p.kernels[0]
    assert "65535" in k.grid
    assert "tl.num_programs(0)" in k.source
    body = ast.parse(k.source).body[0].body
    assert len(body) == 1 and isinstance(body[0], ast.For)
    # The offsets must be derived from the loop tile, not from program_id alone.
    assert "tl.program_id(0)" in k.source and "tile * BLOCK" in k.source
    assert "mask" in k.source
    ast.parse(module([p]))


def test_the_grid_cap_visits_every_tile_exactly_once():
    """The stride is the number of launched programs, so no tile is skipped."""
    def visited(tiles, grid):
        return sorted(t for pid in range(grid) for t in range(pid, tiles, grid))

    for tiles in (1, 2, 3, 65534, 65535, 65536, 131071, 200000):
        grid = min(tiles, 65535)
        assert grid <= 65535
        assert visited(tiles, grid) == list(range(tiles))


def test_platforms_without_a_measured_grid_ceiling_keep_the_direct_launch():
    for name in ("cuda", "kunlunxin", "kunlun-r300"):
        assert resolve_platform(name).max_grid_dim is None
        p = program(resolve_platform(name, mapping_strategy="direct_grid"))
        assert "tl.num_programs(0)" not in p.kernels[0].source
        assert "65535" not in p.kernels[0].grid


@pytest.mark.parametrize("value", [0, -1, True, 65535.0, "65535"])
def test_invalid_grid_ceiling_is_rejected(value):
    with pytest.raises(ValueError, match="max_grid_dim"):
        resolve_platform("ascend", max_grid_dim=value)


ROWWISE = """pipeline rows {
    input x : fp32[R,W]
    s = sum_last(x)
    y = div(x,broadcast_like(s,x))
    output y
}"""


KINDS = ("pointwise", "primitive", "layout")


def rowwise(config):
    return compile_source(ROWWISE, platform_config=config).pipelines[0].triton


def _row_kernel(platform, **kw):
    """The kernel whose grid is the row count itself, not a tile count."""
    p = rowwise(resolve_platform(platform, mapping_strategy="direct_grid", block_size=1024,
                                 allowed_kernel_kinds=KINDS, **kw))
    for k in p.kernels:
        if "cdiv" not in k.grid:
            return p, k
    raise AssertionError([k.grid for k in p.kernels])


def test_a_row_per_program_grid_is_capped_on_ascend():
    """The row-wise primitive launches one program per row, with no tile to shrink.

    Ascend rejects ``coreDim`` above 65535. Unlike the flat pointwise path, the
    row grid has no block size to divide by, so a ``[65536, W]`` RMSNorm input
    asks for 65536 programs however small the reduction tile is. Batch-10
    Tasks 147-150 are all row-wise RMSNorm variants.
    """
    p, k = _row_kernel("ascend")
    assert "65535" in k.grid
    assert "tl.num_programs(0)" in k.source
    body = ast.parse(k.source).body[0].body
    assert len(body) == 1 and isinstance(body[0], ast.For)
    assert "for row in range(tl.program_id(0)" in k.source
    ast.parse(module([p]))


def test_every_row_is_visited_exactly_once_under_the_row_cap():
    """The stride is the number of launched programs, so no row is skipped."""
    def visited(rows, grid):
        return sorted(r for pid in range(grid) for r in range(pid, rows, grid))

    for rows in (1, 2, 3, 65534, 65535, 65536, 65537, 131072, 200000):
        grid = min(rows, 65535)
        assert grid <= 65535
        assert visited(rows, grid) == list(range(rows))


def test_the_row_cap_keeps_the_accumulator_inside_the_loop():
    """Each row must restart its scalar accumulator; a hoisted init would sum rows."""
    _, k = _row_kernel("ascend")
    outer = ast.parse(k.source).body[0].body[0]
    assert isinstance(outer, ast.For)
    inits = [n for n in ast.walk(outer)
             if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
             and getattr(n.value.func, "attr", "") == "full"]
    assert inits, k.source
    # The row index, not a tile offset, is what the store addresses.
    assert "tl.store(s_ptr + row" in k.source


def test_the_row_cap_leaves_the_reduction_tile_unchanged():
    """Only the launch geometry changes; the streaming chunk stays as compiled."""
    _, capped = _row_kernel("ascend")
    _, plain = _row_kernel("kunlunxin")
    tile = "tl.arange(0, 128)"
    assert tile in capped.source and tile in plain.source
    assert "tl.cdiv(W, 128)" in capped.source and "tl.cdiv(W, 128)" in plain.source


def test_rowwise_targets_without_a_ceiling_keep_one_program_per_row():
    for name in ("cuda", "kunlunxin", "kunlun-r300"):
        p = rowwise(resolve_platform(name, mapping_strategy="direct_grid", block_size=1024,
                                     allowed_kernel_kinds=KINDS))
        for k in p.kernels:
            assert "tl.num_programs(0)" not in k.source, (name, k.source)
            assert "65535" not in k.grid, (name, k.grid)


RMSNORM = """pipeline rms {
    input x : bf16[B,D]
    input weight : bf16[D]
    y = rmsnorm_hf(x, weight)
    output y
}"""

FUSED_REDUCTION = """pipeline softmax {
    input x : fp32[B,S]
    m = max_last(x)
    e = exp(sub(x,broadcast_like(m,x)))
    z = sum_last(e)
    o = div(e,broadcast_like(z,x))
    output o
}"""

ALL_KINDS = ("pointwise", "rowwise", "matmul", "transpose", "rope", "layout",
             "primitive", "cache_store", "linear_decode")


def kernels(source, platform, kinds=ALL_KINDS):
    config = resolve_platform(platform, mapping_strategy="direct_grid", block_size=1024,
                              allowed_kernel_kinds=kinds)
    return compile_source(source, platform_config=config).pipelines[0].triton.kernels


def test_a_rowwise_rmsnorm_grid_is_capped_on_ascend():
    """Batch-10 Tasks 147-150 are all row-wise RMSNorm variants.

    ``rmsnorm_hf`` lowers to a single kernel that gives every program one row, so
    its grid is the row count. ``B`` above 65535 is an ordinary batch-times-
    sequence size, and Ascend rejects that ``coreDim`` outright.
    """
    (k,) = kernels(RMSNORM, "ascend", ("rowwise",))
    assert k.kind == "rowwise"
    assert "65535" in k.grid and "tl.num_programs(0)" in k.source
    assert "for row in range(tl.program_id(0)" in k.source


def test_the_fused_reduction_path_is_capped_on_ascend():
    """The softmax-shaped reduction group launches on the row grid as well."""
    ks = kernels(FUSED_REDUCTION, "ascend")
    assert ks, "the softmax pipeline must lower"
    for k in ks:
        assert "65535" in k.grid, (k.kind, k.grid)


def test_the_reduction_cap_leaves_the_row_count_unchanged():
    """Only the launch geometry changes: the loop bound is still the row count."""
    (k,) = kernels(RMSNORM, "ascend", ("rowwise",))
    ast.parse(k.source)
    loop = ast.parse(k.source).body[0].body[0]
    assert isinstance(loop, ast.For)
    assert isinstance(loop.iter, ast.Call)
    assert getattr(loop.iter.func, "id", "") == "range"
    # Every row is still visited: the loop counts to `rows`, striding by the
    # number of programs the capped launch actually started.
    stride = loop.iter.args[2]
    assert isinstance(stride, ast.Call)
    assert getattr(stride.func, "attr", "") == "num_programs"


def test_no_ascend_row_grid_is_launched_uncapped():
    """Corpus invariant: the row-wise kind is capped wherever a ceiling exists."""
    import pathlib
    sources = [p.read_text() for p in
               list(pathlib.Path("examples").rglob("*.fgpu"))
               + list(pathlib.Path("test").rglob("*.fgpu"))]
    assert sources
    checked = 0
    for source in sources:
        try:
            ks = kernels(source, "ascend")
        except Exception:
            continue
        for k in ks:
            if k.kind != "rowwise":
                continue
            checked += 1
            assert "65535" in k.grid, (k.grid, source[:60])
    assert checked, "no row-wise kernel in the corpus: the invariant proved nothing"


def test_rowwise_targets_without_a_ceiling_keep_the_plain_row_grid():
    for name in ("cuda", "kunlunxin", "kunlun-r300"):
        for k in kernels(RMSNORM, name, ("rowwise", "primitive", "pointwise")): 
            assert "tl.num_programs(0)" not in k.source, (name, k.kind)
            assert "65535" not in k.grid, (name, k.kind, k.grid)


def test_the_literal_row_form_keeps_its_program_index():
    """A literal row length takes the single-tile form on every platform.

    Capping drops the ``row = tl.program_id(0)`` assignment because the loop
    header replaces it. A platform without a ceiling must therefore keep the
    assignment, or the kernel references an undefined index -- this is a
    regression that was introduced and caught here.
    """
    literal = """pipeline lit {
        input x : fp32[B,1024]
        input weight : fp32[1024]
        y = rmsnorm_hf(x, weight)
        output y
    }"""
    for platform in ("cuda", "kunlunxin", "ascend"):
        for k in kernels(literal, platform, ("rowwise",)):
            assigned = k.source.count("row = tl.program_id(0)")
            looped = k.source.count("for row in range(tl.program_id(0)")
            assert assigned + looped == 1, (platform, k.grid, k.source)


def test_every_compute_kernel_defines_its_program_index_exactly_once():
    """The cap rewrites every program_id to the loop tile, never duplicating it."""
    import pathlib
    corpus = [p.read_text() for p in
              list(pathlib.Path("examples").rglob("*.fgpu"))
              + list(pathlib.Path("test").rglob("*.fgpu"))]
    checked = 0
    for source in corpus:
        for platform in ("cuda", "kunlunxin", "ascend"):
            try:
                ks = kernels(source, platform)
            except Exception:
                continue
            for k in ks:
                if not str(k.grid):
                    continue          # a pure layout kernel launches nothing
                checked += 1
                assert k.source.count("tl.program_id(0)") == 1, (platform, k.kind, k.grid)
    assert checked > 100, checked


MATMUL = """pipeline mm {
    input a : fp16[M,K]
    input b : fp16[K,N]
    y = matmul(a,b)
    output y
}"""

TRANSPOSE = """pipeline tr {
    input x : fp16[M,N]
    y = transpose(x)
    output y
}"""


def test_a_two_dimensional_matmul_grid_is_capped_on_ascend():
    """Ascend refuses a launch whose *program count* exceeds 65535.

    The matmul grid is (cdiv(M,BLOCK_M), cdiv(N,BLOCK_N)). The ceiling bounds
    the count, not either axis, so the axes are capped at floor(sqrt(65535)) =
    255 each: capping both at 65535 would allow 65535**2 programs, and the
    official Task 144 evaluation of 2026-10-19 was rejected with
    ``Invalid_Argument(EE1003): coreDim 78336`` for exactly that reason. Each
    axis stays strided by the programs actually launched.
    """
    cap = isqrt(65535)
    for platform in ("ascend",):
        for k in kernels(MATMUL, platform, ("matmul",)):
            # one cap per axis, each written as a comparison and an else branch
            assert k.grid.count(f"else {cap}") == 2, k.grid
            assert "65535" not in k.grid, k.grid
            assert k.source.count("tl.num_programs(") == 2, k.source
            assert "for pid_m in range(tl.program_id(0)" in k.source
            assert "for pid_n in range(tl.program_id(1)" in k.source
            ast.parse(k.source)


def test_the_swizzled_matmul_grid_is_capped_on_ascend():
    """The L2-swizzled form launches on one axis holding cdiv(M)*cdiv(N) blocks."""
    config = resolve_platform("ascend", mapping_strategy="direct_grid", block_size=1024,
                              allowed_kernel_kinds=("matmul",), group_m=8)
    (k,) = compile_source(MATMUL, platform_config=config).pipelines[0].triton.kernels
    assert "65535" in k.grid
    assert "for tile in range(tl.program_id(0)" in k.source
    # Inside the kernel the block count must be a device expression.
    assert "tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)" in k.source


def test_a_two_dimensional_transpose_grid_is_capped_on_ascend():
    cap = isqrt(65535)
    (k,) = kernels(TRANSPOSE, "ascend", ("transpose",))
    assert k.grid.count(f"else {cap}") == 2, k.grid
    assert "65535" not in k.grid, k.grid
    assert k.source.count("tl.num_programs(") == 2
    ast.parse(k.source)


def _eval_launch(kernel, **names):
    """Evaluate a generated launch expression against stand-in names."""
    class _Triton:
        @staticmethod
        def cdiv(a, b):
            return (a + b - 1) // b

    env = {"triton": _Triton()}
    env.update(names)
    return eval(kernel.grid, env)          # a generated literal, no locals


def test_the_emitted_two_dimensional_grid_bounds_the_program_count():
    """The ceiling bounds programs launched, so the *product* of the axes must fit.

    An axis-by-axis cap is not a bound: two axes capped at the ceiling ask for
    ceiling**2 programs. The official Task 144 evaluation of 2026-10-19 was
    rejected with ``Invalid_Argument(EE1003): coreDim 78336`` for a grid whose
    axes were all small. This evaluates the expression that is actually
    emitted, so a regression in the emitter cannot pass by reimplementing the
    rule here.
    """
    cap = isqrt(65535)
    (k,) = kernels(TRANSPOSE, "ascend", ("transpose",))
    for m, n in ((1, 1), (128, 128), (cap * 128, cap * 128), (256 * 128, 256 * 128),
                 (65535 * 128, 3), (3, 65535 * 128), (10**7, 10**7)):
        launched = _eval_launch(k, M=m, N=n, BLOCK_M=128, BLOCK_N=128)
        assert len(launched) == 2, launched
        assert all(value >= 1 for value in launched), (m, n, launched)
        assert launched[0] * launched[1] <= 65535, (m, n, launched)
        assert launched[0] <= cap and launched[1] <= cap, (m, n, launched)


def test_a_degenerate_two_dimensional_launch_does_not_divide_by_zero():
    """A zero tile count must not crash the host: the cap is a constant.

    Capping the second axis at ``ceiling // rows_launched`` would divide by zero
    for a degenerate shape, which the previous axis-by-axis cap did not do.
    """
    cap = isqrt(65535)
    (k,) = kernels(TRANSPOSE, "ascend", ("transpose",))
    for m, n in ((0, 0), (0, 7), (7, 0)):
        rows, columns = _eval_launch(k, M=m, N=n, BLOCK_M=128, BLOCK_N=128)
        assert 0 <= rows <= cap and 0 <= columns <= cap, (m, n, rows, columns)


def test_every_row_and_column_block_is_visited_exactly_once():
    def visited(blocks, launched):
        return sorted(b for pid in range(launched) for b in range(pid, blocks, launched))

    cap = isqrt(65535)
    for blocks in (1, 2, 254, 255, 256, 65536, 131072):
        launched = min(blocks, cap)
        assert visited(blocks, launched) == list(range(blocks))
    # And the two-axis case decomposes into two independent 1-D coverages.
    for m_blocks, n_blocks in ((65536, 3), (3, 65536), (70000, 70000)):
        for blocks in (m_blocks, n_blocks):
            assert visited(blocks, min(blocks, cap)) == list(range(blocks))


def test_rectangular_targets_without_a_ceiling_keep_their_exact_grid():
    for platform in ("cuda", "kunlunxin"):
        for k in kernels(MATMUL, platform, ("matmul",)):
            assert "65535" not in k.grid, (platform, k.grid)
            assert "tl.num_programs(" not in k.source, (platform, k.kind)
