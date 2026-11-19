"""Regression tests for the 2026-10-26 re-review findings.

The second review found six items the first round had not actually fixed plus
four regressions introduced by the repair.  Every test here fails on the
commit the review examined and passes now (each one was reproduced first).
"""
import contextlib
import io
import tempfile
from pathlib import Path

import pytest

from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.frontend.diagnostic import CompileError
from flowgpu.compiler.cli import main


HERE = Path(__file__).resolve().parents[1]


def _kernel(source, index=0):
    pipeline = compile_source(source).pipelines[0]
    assert pipeline.triton.complete, [r.message() for r in pipeline.triton.refusals]
    return pipeline.triton.kernels[index]


def _formal_names(signature):
    parameters = signature.split("(")[1].rstrip("):")
    return [part.split(":")[0].strip() for part in parameters.split(",") if part.strip()]


# --- new regression: transpose + scalar epilogue crashed -------------------
def test_transpose_with_a_scalar_epilogue_compiles():
    kernel = _kernel(
        "pipeline p { input x:bf16[4,8] input s:fp32\n"
        "             t = transpose(x) y = mul(t, s) output y }")
    assert kernel.signature.startswith("def p_k0(x_ptr")
    names = _formal_names(kernel.signature)
    assert len(names) == len(set(names))
    assert "s" in names                      # the scalar is a formal...
    assert "t0 * s" in kernel.source         # ...read directly by the epilogue


# --- P1-2: scalar names must not collide with dimension symbols ------------
@pytest.mark.parametrize("source,renamed,dimension_expected", [
    # a multi-dim kernel passes the dimension symbols it indexes with
    ("pipeline p { input a:fp32[M,K] input b:fp32[K,N] input M:fp32 "
     "c = matmul(a, b) d = mul(c, M) output d }", "M_2", True),
    ("pipeline p { input x:fp32[M,N] input N:fp32 "
     "z = softmax(x) y = mul(z, N) output y }", "N_2", True),
    # the flat pointwise kernel addresses with `numel`, so the bare dimension
    # never becomes a formal there -- the scalar still gets its own name
    ("pipeline p { input x:fp32[M,4] input M:fp32 y = mul(x, M) output y }",
     "M_2", False),
    ("pipeline p { input x:fp32[M,N] input N:fp32 "
     "z = sum_last(x) y = mul(z, N) output y }", "N_2", True),
])
def test_a_scalar_named_like_a_dimension_gets_its_own_formal(
        source, renamed, dimension_expected):
    kernel = _kernel(source)
    names = _formal_names(kernel.signature)
    assert len(names) == len(set(names)), names
    assert renamed in names, (renamed, names)
    dimension = renamed.split("_")[0]
    if dimension_expected:
        # the bare name stays the dimension, and the body uses it as one
        assert dimension in names, names
    assert renamed in kernel.source


def test_an_input_named_like_a_builtin_does_not_shadow_the_host():
    pipeline = compile_source(
        "pipeline p { input float : fp32[4] y = relu(float) output y }").pipelines[0]
    assert pipeline.triton.host.splitlines()[0] == "def run_p(float_2):"


# --- new regression: cast_to crashed the default CLI output ----------------
@pytest.mark.parametrize("emit", [None, "json", "ast", "summary", "reference", "triton"])
def test_a_cast_to_program_renders_in_every_emit_mode(emit, tmp_path):
    path = tmp_path / "cast.fgpu"
    path.write_text("pipeline p { input x:fp32[4] y = cast_to(x, bf16) output y }\n")
    stream, errors = io.StringIO(), io.StringIO()
    argv = [str(path)] if emit is None else [str(path), "--emit", emit]
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors):
        code = main(argv)
    assert code == 0, (emit, errors.getvalue())


# --- P1-5: store_cache keeps its rank diagnostics --------------------------
def test_store_cache_rank_one_view_is_a_diagnostic():
    with pytest.raises(CompileError, match='olumn 9: store_cache_4d views must be rank 4') as caught:
        compile_source(
            "pipeline sc { input k_view:fp16[128] input v_view:fp16[128] "
            "input cache_k:fp16[4,2,3,3] input cache_v:fp16[4,2,3,3] "
            "input loc:int64[4]\n"
            " k, v = store_cache_4d(k_view, v_view, cache_k, cache_v, loc, 2)\n"
            " output k\n output v }")
    assert "rank 4" in str(caught.value)


# --- P1-6: a dead-code skip is not a compile failure ----------------------
def _cli(argv):
    stream, errors = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(errors):
        code = main(argv)
    return code, stream.getvalue(), errors.getvalue()


def test_a_dead_intermediate_is_not_a_reference_failure(tmp_path):
    path = tmp_path / "dead.fgpu"
    path.write_text("pipeline p { input x:fp32[8] y = mul(x, x) "
                    "z = relu(x) output z }\n")
    for argv in ([str(path), "--emit", "triton"],
                 [str(path), "--emit", "reference"], [str(path), "--run"]):
        code, _out, err = _cli(argv)
        assert code == 0, (argv, err)


# --- dead epilogue on a scan is dead code, not INTERNAL --------------------
def test_a_dead_scan_epilogue_is_eliminated():
    pipeline = compile_source(
        "pipeline p { input x:fp32[2,8] y,t = cumsum_totals(x) "
        "w = mul(t,t) output y }").pipelines[0]
    assert pipeline.triton.complete
    source = pipeline.triton.kernels[0].source
    assert "tl.store(y_ptr" in source
    assert "w" not in [value for value in pipeline.memory.buffers if value.storage]


def test_a_dead_scan_epilogue_next_to_a_live_totals_result():
    pipeline = compile_source(
        "pipeline p { input x:fp32[2,8] y,t = cumsum_totals(x) "
        "w = mul(t,t) output y output t }").pipelines[0]
    assert pipeline.triton.complete
    source = pipeline.triton.kernels[0].source
    assert "tl.store(y_ptr" in source and "tl.store(t_ptr" in source


# --- structural fixes that had no tests -----------------------------------
def test_emit_false_stops_before_both_lowerings():
    result = compile_source("pipeline p { input x:fp32[4] y = relu(x) output y }",
                            emit=False)
    pipeline = result.pipelines[0]
    assert pipeline.triton is None
    assert pipeline.reference is None


def test_program_scalar_identifiers_reject_python_keywords():
    from flowgpu.compiler.ir.program_scalar_ir import ProgramIndex, ProgramScalar
    with pytest.raises(ValueError, match='program scalar table must be a non-empty, non-keyword identifier') as caught:
        ProgramScalar(table="class", index=ProgramIndex(axis=0))
    assert "keyword" in str(caught.value)


def test_the_cache_facade_does_not_rewrite_the_implementation_fingerprint():
    from flowgpu.compiler import cache
    from flowgpu.compiler.runtime import cache as implementation

    original = implementation.compiler_fingerprint
    calls = []

    def replacement():
        calls.append(True)
        return "deadbeef" * 8

    cache.compiler_fingerprint = replacement
    try:
        with tempfile.TemporaryDirectory() as directory:
            source = "pipeline p { input x:fp32[4] y = relu(x) output y }"
            cache.cached_module(source, cache_dir=directory)
        assert calls                      # the facade's value was used
        assert implementation.compiler_fingerprint is original   # not swapped
    finally:
        cache.compiler_fingerprint = original


# --- DCE ordering: a dead group is skipped before any refusal --------------
@pytest.mark.parametrize("source", [
    "pipeline p { input x:fp32[2,8] input r:fp32[2,1]\n"
    "             y = add_broadcast(x, r) z = relu(x) output z }",
    "pipeline p { input x:fp32[2,8] input r:fp32[2,1]\n"
    "             c = cumsum(x) y = add_broadcast(c, r) z = relu(c) output z }",
    "pipeline p { input x:fp32[2,8]\n"
    "             a,b = cumsum_totals(x) y = add_broadcast(a, b) output b }",
])
def test_a_dead_group_is_eliminated_before_the_refusal_classification(source):
    pipeline = compile_source(source).pipelines[0]
    assert pipeline.triton.complete
    assert not pipeline.triton.refusals
    assert pipeline.triton.kernels


def test_the_reference_host_and_kernel_reserve_range_and_the_other_builtins():
    pipeline = compile_source(
        "pipeline p { input range:fp32[4] y = relu(range) output y }").pipelines[0]
    assert pipeline.reference.host.splitlines()[0] == "def run_p(range_2):"
    assert pipeline.triton.host.splitlines()[0] == "def run_p(range_2):"
    # The kernel function shadows builtins too (it calls range/len/float), so
    # its parameter must be renamed as well; execute the reference program to
    # prove the body and the formal agree.
    signature = next(line for line in pipeline.reference.kernels[0].source.splitlines()
                     if line.startswith("def "))
    assert "range_value" in signature
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    entry = next(value for name, value in namespace.items() if name.startswith("run_"))
    assert entry([-1.0, 2.0, -3.0, 4.0])["y"] == [0.0, 2.0, 0.0, 4.0]


# --- rowwise reductions stay lane-shaped for the strict Triton forks ------
@pytest.mark.parametrize("source, marker", [
    ("pipeline p { input x:bf16[2,64] input weight:bf16[64]\n"
     "             y = rmsnorm_hf(x, weight) output y }", "tl.rsqrt"),
    ("pipeline p { input x:bf16[4,64] y = softmax(x) output y }", "tl.exp"),
])
def test_rowwise_reduction_results_are_broadcast_lane_shaped(source, marker):
    """A reduction result must enter vector arithmetic already lane-shaped.

    Several of the seven FlagOS competition Triton forks reject 0-d result <->
    vector arithmetic, so the emitter wraps the scalar in ``offsets * 0.0``.
    """
    kernel = compile_source(source).pipelines[0].triton.kernels[0]
    assert marker in kernel.source
    assert "offsets * 0.0 +" in kernel.source


# --- GEMM tile policy: measured tiles are data, not emitter edits ---------
def test_the_matmul_tile_policy_rewrites_header_and_launch_resources():
    from flowgpu.compiler.backend.platform_configs import resolve_platform
    from flowgpu.compiler.backend.triton_backend import module
    source = (HERE / 'examples/gpu_mode/matmul.fgpu').read_text()
    config = resolve_platform('cuda', matmul_block_m=128, matmul_block_n=256,
                              matmul_block_k=32, matmul_num_warps=8,
                              matmul_num_stages=4, group_m=8)
    program = compile_source(source, platform_config=config).pipelines[0].triton
    text = module([program])
    assert 'BLOCK_M = 128' in text and 'BLOCK_N = 256' in text
    assert 'BLOCK_K = 32' in text
    assert 'num_warps=8' in text and 'num_stages=4' in text
    # The default is now the measured CUDA floor (2026-10-27); the historical
    # 64x64x32 bytes are reproducible by pinning the legacy fields explicitly.
    default = module([compile_source(source).pipelines[0].triton])
    assert 'BLOCK_M = 128' in default and 'BLOCK_N = 128' in default
    assert 'BLOCK_K = 64' in default and 'num_stages=3' in default
    legacy = resolve_platform('cuda', matmul_block_m=64, matmul_block_n=64,
                              matmul_block_k=32, matmul_num_warps=4,
                              matmul_num_stages=2)
    old = module([compile_source(source, platform_config=legacy).pipelines[0].triton])
    assert 'BLOCK_M = 64' in old and 'num_stages=2' in old
