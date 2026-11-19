"""The chunked two-pass lowering for rows longer than one reduction tile.

v0.1 assumed the reduced axis fits a single block; a longer row compiled to
a tile the target could not sustain, which is wrong rather than slow. These
tests pin the replacement: a platform with a measured reduction budget
(kunlunxin, from the uni_sram evidence) lowers symbolic or over-budget rows
to two sweeps, while a platform without one keeps the v0.1 form instead of
inventing a ceiling.
"""

import pytest

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.platform_configs import resolve_platform

SOFTMAX_D = """pipeline soft {
    input x : bf16[N,D]
    y = softmax(x)
    output y
}"""

RMS_ADD = """pipeline fused {
    input x : bf16[N,D]
    input r : bf16[N,D]
    input w : bf16[D]
    t = add(x, r)
    y = rms_norm(t, w)
    output y
}"""

SOFTMAX_LITERAL_LONG = """pipeline wide {
    input x : bf16[4,4096]
    y = softmax(x)
    output y
}"""

SOFTMAX_LITERAL_SHORT = """pipeline narrow {
    input x : bf16[4,64]
    y = softmax(x)
    output y
}"""

KUNLUN_ROWWISE = resolve_platform(
    "kunlunxin", allowed_kernel_kinds=("pointwise", "rowwise")
)


def program(source, config):
    return compile_source(source, platform_config=config).pipelines[0].triton


def kernel(source, config):
    return program(source, config).kernels[0]


# ---- Which form a row takes ----

def test_cuda_with_a_measured_limit_longs_a_symbolic_row():
    """CUDA's limit is measured now (A800, 2026-10-02): a symbolic row
    lowers to the chunked form, capped at the measured 4096."""
    k = kernel(SOFTMAX_D, resolve_platform("cuda"))
    assert "for chunk" in k.source
    assert "tl.arange(0, ROW)" in k.source


def test_cuda_default_emission_stays_single_tile_below_the_limit():
    """The v0.1 single-tile form survives as the default for short rows:
    D=64 needs no chunking even with the measured limit in place."""
    k = kernel(SOFTMAX_LITERAL_SHORT, resolve_platform("cuda"))
    assert "for chunk" not in k.source


def test_kunlun_limit_lowers_a_symbolic_row_to_two_sweeps():
    k = kernel(SOFTMAX_D, KUNLUN_ROWWISE)
    source = k.source
    assert source.count("for chunk in range(0, tl.cdiv(D, ROW))") == 2
    assert "running_max" in source
    assert "running_sum" in source


def test_a_literal_row_over_the_limit_is_chunked():
    k = kernel(SOFTMAX_LITERAL_LONG, KUNLUN_ROWWISE)
    assert "for chunk" in k.source
    # 4096 > 128, so the chunk width is the limit itself, not next_power_of_2.
    assert "128" in program(SOFTMAX_LITERAL_LONG, KUNLUN_ROWWISE).host


def test_a_literal_row_within_the_limit_stays_single_tile():
    k = kernel(SOFTMAX_LITERAL_SHORT, KUNLUN_ROWWISE)
    assert "for chunk" not in k.source
    assert "tl.arange(0, ROW)" in k.source


def test_the_chunk_width_is_capped_at_launch_time():
    """A symbolic row picks min(next_power_of_2(D), limit) at launch."""
    host = program(SOFTMAX_D, KUNLUN_ROWWISE).host
    assert "triton.next_power_of_2(D) if triton.next_power_of_2(D) <= 128 else 128" in host


# ---- The reduction is folded correctly across sweeps ----

def test_softmax_running_maximum_rescales_the_running_sum():
    """The online-softmax invariant: sum is always expressed in the current
    maximum's scale, so every chunk contributes its exp(x - new_max)."""
    source = kernel(SOFTMAX_D, KUNLUN_ROWWISE).source
    assert "tl.max(tl.where(mask, " in source
    assert "tl.exp(running_max - t3)" in source
    assert "running_max = t3" in source
    # The second sweep divides by the finished sum, not a chunk-local one.
    assert "t5 / running_sum" in source
    # The maximum starts at -inf so the first chunk can only raise it;
    # starting at 0 would silently floor every negative row.
    assert "tl.full([], -float('inf'), tl.float32)" in source


def test_each_sweep_recomputes_offsets_and_mask_before_loading():
    """A sweep that drops its mask assignment would reference an undefined
    variable, or silently drop padding exclusion: every load must sit after
    its own sweep's offsets and mask assignments, inside the loop."""
    source = kernel(SOFTMAX_D, KUNLUN_ROWWISE).source
    assert source.count("mask = offsets < D") == 2
    loops = []
    start = 0
    while True:
        at = source.find("for chunk in range(0, tl.cdiv(D, ROW))", start)
        if at < 0:
            break
        loops.append(at)
        start = at + 1
    for loop in loops:
        offsets_at = source.index("offsets = chunk * ROW + tl.arange(0, ROW)", loop)
        mask_at = source.index("mask = offsets < D", loop)
        first_load = source.index("tl.load(", loop)
        assert loop < offsets_at < mask_at < first_load


def test_the_store_is_inside_the_second_sweep():
    """Storing outside the loop would write only the last chunk's tile; the
    store must be an 8-space-indented statement, i.e. inside the loop body."""
    source = kernel(SOFTMAX_D, KUNLUN_ROWWISE).source
    store_lines = [line for line in source.splitlines() if "tl.store(" in line]
    assert len(store_lines) == 1
    assert store_lines[0].startswith("        tl.store(")


def test_softmax_padding_never_enters_either_sweep():
    source = kernel(SOFTMAX_D, KUNLUN_ROWWISE).source
    # The chunk maximum reads padding as -inf ...
    assert "tl.where(mask, t1, -float('inf'))" in source
    # ... and the running sum adds nothing for padding lanes.
    assert "tl.where(mask, tl.exp(t1 - t3), 0.0)" in source
    # The stored chunk divides and masks with the same offsets.
    assert "mask=mask)" in source


def test_rms_norm_squares_fold_across_chunks():
    source = kernel(RMS_ADD, KUNLUN_ROWWISE).source
    assert "sum_of_squares += tl.sum(" in source
    assert "mean_of_squares = sum_of_squares / D" in source
    assert "tl.rsqrt(mean_of_squares + 1e-06)" in source


def test_a_fused_prologue_is_recomputed_in_both_sweeps():
    """The add -> rms_norm fusion must survive chunking: the prologue is pure,
    so the second sweep recomputes it rather than reading a buffer that the
    fusion eliminated."""
    source = kernel(RMS_ADD, KUNLUN_ROWWISE).source
    assert source.count("(t0 + t1)    # t: fused") == 1
    assert source.count("(t4 + t5)    # t: fused") == 1
    # The weight is a second-sweep operand only: loading it in sweep one
    # would read past the row's useful data for no reason.
    assert "tl.load(w_ptr" in source
    assert source.index("tl.load(w_ptr") > source.index("mean_of_squares")


def test_the_rowwise_grid_is_still_one_program_per_row():
    assert kernel(SOFTMAX_D, KUNLUN_ROWWISE).grid == "(N,)"


def test_a_chunked_rms_norm_still_disables_fp_fusion():
    assert kernel(RMS_ADD, KUNLUN_ROWWISE).launch_options == (
        ("enable_fp_fusion", False),
    )


# ---- The ceiling is a platform contract ----

@pytest.mark.parametrize("kwargs", [
    {"reduction_block_limit": 0},
    {"reduction_block_limit": -128},
    {"reduction_block_limit": 100},
    {"reduction_block_limit": True},
])
def test_invalid_reduction_limits_are_rejected(kwargs):
    with pytest.raises(ValueError, match="reduction_block_limit"):
        resolve_platform("kunlunxin", **kwargs)


def test_kunlunxin_profiles_carry_the_measured_limit():
    assert resolve_platform("kunlunxin").reduction_block_limit == 128
    assert resolve_platform("kunlun-r300").reduction_block_limit is None
    assert resolve_platform("cuda").reduction_block_limit == 4096


def test_the_limit_can_be_overridden():
    config = resolve_platform(
        "kunlunxin", reduction_block_limit=64,
        allowed_kernel_kinds=("pointwise", "rowwise"),
    )
    assert config.reduction_block_limit == 64
    host = program(SOFTMAX_D, config).host
    assert "<= 64 else 64" in host


def test_the_limit_appears_in_emitted_metadata():
    meta = program(SOFTMAX_D, KUNLUN_ROWWISE).platform_config.as_dict()
    assert meta["reduction_block_limit"] == 128


# ---- CLI surface ----

def test_cli_emits_the_chunked_form_with_a_limit(tmp_path, capsys):
    path = tmp_path / "soft.fgpu"
    path.write_text(SOFTMAX_D)
    from flowgpu.compiler.cli import main
    assert main(["--emit", "triton", str(path), "--reduction-block-limit", "256"]) == 0
    out = capsys.readouterr().out
    assert out.count("for chunk in range(0, tl.cdiv(D, ROW))") == 2
    assert "<= 256 else 256" in out


def test_cli_rejects_a_non_power_of_two_limit(tmp_path, capsys):
    path = tmp_path / "soft.fgpu"
    path.write_text(SOFTMAX_D)
    from flowgpu.compiler.cli import main
    assert main(["--emit", "triton", str(path), "--reduction-block-limit", "300"]) == 2
    assert "reduction_block_limit" in capsys.readouterr().err


def test_cli_default_cuda_emission_is_chunked_for_symbolic_rows(tmp_path, capsys):
    """The default CUDA profile now carries the A800-measured limit, so a
    symbolic row lowers to the chunked form without any flag."""
    path = tmp_path / "soft.fgpu"
    path.write_text(SOFTMAX_D)
    from flowgpu.compiler.cli import main
    assert main(["--emit", "triton", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.count("for chunk in range(0, tl.cdiv(D, ROW))") == 2
    assert "<= 4096 else 4096" in out


def test_cli_can_raise_the_ceiling(tmp_path, capsys):
    """--reduction-block-limit overrides the measured default: a symbolic
    row then chunks at the caller's width instead of 4096."""
    path = tmp_path / "soft.fgpu"
    path.write_text(SOFTMAX_D)
    from flowgpu.compiler.cli import main
    assert main(["--emit", "triton", str(path),
                 "--reduction-block-limit", "8192"]) == 0
    out = capsys.readouterr().out
    assert out.count("for chunk in range(0, tl.cdiv(D, ROW))") == 2
    assert "<= 8192 else 8192" in out
