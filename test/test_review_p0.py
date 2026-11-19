"""Regression tests for the three silent-wrong-answer defects found in the
2026-10-26 code review (report P0-1..P0-3).

Each test pins the *observable* contract, not the fix's shape: a live result
must be written by an emitted kernel, a bool output must be stored, and a
literal `segmented_matmul` must cover every column even when N exceeds the
kernel's lane block.
"""
import math
import sys
import pathlib

import pytest

from compiler.backend import reference
from compiler.backend.triton_backend import module as triton_module
from compiler.compile import compile_source
from compiler.frontend.diagnostic import CompileError
from test import segmented_support

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def _compile(source):
    return compile_source(source).pipelines[0]


def _reference_run(pipeline, *inputs):
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    name = next(key for key in namespace if key.startswith("run_"))
    return namespace[name](*inputs)


# ---------------------------------------------------------------------------
# P0-1: a partially dead multi-result group must still produce its live result.
def test_cumsum_totals_keeps_the_live_values_result():
    pipeline = _compile(
        "pipeline p { input x : fp32[4,8]\n"
        "             a, b = cumsum_totals(x)\n"
        "             output a }")
    kernels = pipeline.triton.kernels
    assert len(kernels) == 1, [kernel.ops for kernel in kernels]
    assert not pipeline.triton.skipped, pipeline.triton.skipped
    source = kernels[0].source
    assert "tl.store(a_ptr" in source
    assert "b_ptr" not in source
    produced = _reference_run(pipeline, list(range(1, 33)))["a"]
    expected = []
    for row in range(4):
        running = 0
        for column in range(8):
            running += row * 8 + column + 1
            expected.append(running)
    assert [round(value, 6) for value in produced] == [float(value) for value in expected]


def test_cumsum_totals_keeps_the_live_totals_result():
    pipeline = _compile(
        "pipeline p { input x : fp32[4,8]\n"
        "             a, b = cumsum_totals(x)\n"
        "             output b }")
    kernels = pipeline.triton.kernels
    assert len(kernels) == 1, [kernel.ops for kernel in kernels]
    assert not pipeline.triton.skipped, pipeline.triton.skipped
    source = kernels[0].source
    assert "tl.store(b_ptr" in source
    assert "a_ptr" not in source
    produced = _reference_run(pipeline, list(range(1, 33)))["b"]
    expected = [sum(range(row * 8 + 1, row * 8 + 9)) for row in range(4)]
    assert [round(value, 6) for value in produced] == [float(value) for value in expected]


def test_a_fully_dead_multi_result_group_is_dead_code():
    """The other side of the rule: nothing live means nothing to emit."""
    pipeline = _compile(
        "pipeline p { input x : fp32[4,8]\n"
        "             a, b = cumsum_totals(x)\n"
        "             y = mul(x, x)\n"
        "             output y }")
    assert all("cumsum_totals" not in kernel.ops for kernel in pipeline.triton.kernels)
    assert pipeline.triton.skipped


# ---------------------------------------------------------------------------
# P0-2: a bool result must be stored, not merely computed.
def test_a_bool_primitive_result_is_stored():
    pipeline = _compile(
        "pipeline p { input x : fp32[4]\n"
        "             input y : fp32[4]\n"
        "             m = lt(x, y)\n"
        "             output m }")
    kernels = pipeline.triton.kernels
    assert len(kernels) == 1
    source = kernels[0].source
    assert "tl.store(m_ptr" in source
    values = [0.0, 1.0, 2.0, 3.0]
    other = [1.0, 0.0, 3.0, 2.0]
    result, _ = segmented_support.execute(
        triton_module([pipeline.triton]),
        (segmented_support.Buffer(values, (4,), "fp32"),
         segmented_support.Buffer(other, (4,), "fp32")),
        pipeline="p")
    assert result["m"] == [True, False, True, False]


# ---------------------------------------------------------------------------
# P0-3: literal segmented_matmul must cover every column (N > lane block).
def test_literal_segmented_matmul_covers_columns_past_the_lane_block():
    rows, reduction, columns = 2, 4, 1100
    source = ("pipeline grouped {\n"
              f" input x:fp32[{rows},{reduction}]\n"
              f" input w:fp32[1,{reduction},{columns}]\n"
              " input indptr:int32[2]\n"
              " out = segmented_matmul(x, w, indptr)\n"
              " output out\n"
              "}\n")
    pipeline = _compile(source)
    x_values = [float((index % 7) - 3) for index in range(rows * reduction)]
    w_values = [float(((index * 13) % 17) - 8) for index in range(reduction * columns)]
    expected = []
    for row in range(rows):
        for column in range(columns):
            total = 0.0
            for k in range(reduction):
                total += x_values[row * reduction + k] * w_values[k * columns + column]
            expected.append(total)
    result, launches = segmented_support.execute(
        triton_module([pipeline.triton]),
        (segmented_support.Buffer(x_values, (rows, reduction), "fp32"),
         segmented_support.Buffer(w_values, (1, reduction, columns), "fp32"),
         segmented_support.Buffer([0, rows], (2,), "int32")),
        pipeline="grouped")
    produced = result["out"]
    # every element written (the CPU double asserts that) and numerically correct
    assert len(produced) == rows * columns
    for got, want in zip(produced, expected):
        assert abs(got - want) <= 1e-4 + 1e-4 * abs(want)


# ---------------------------------------------------------------------------
# P1-1: a scan epilogue and a live scan result cannot share one kernel; the
# pre-fix behaviour was a KeyError in the reference lowering and an unwritten
# result on the device path.
def test_a_scan_epilogue_with_a_live_scan_result_is_refused():
    with pytest.raises(CompileError, match="ipeline 'p': a scan with a fused pointwise epilogue can materialize only the epilogue result; write the other scan result in its own statement \\(live: y\\)") as caught:
        compile_source(
            "pipeline p { input x : fp32[2,8]\n"
            "             y, t = cumsum_totals(x)\n"
            "             w = mul(t, t)\n"
            "             output y\n"
            "             output w }")
    message = str(caught.value)
    assert "fused pointwise epilogue" in message
    assert "own statement" in message


# ---------------------------------------------------------------------------
# P1-12: two pipelines with one name would shadow each other's `run_<name>`.
def test_duplicate_pipeline_names_are_a_compile_error():
    with pytest.raises(CompileError, match="olumn 10: duplicate pipeline name 'clash'; every pipeline in a source file needs a unique name") as caught:
        compile_source(
            "pipeline clash { input x : fp32[4] y = relu(x) output y }\n"
            "pipeline clash { input x : fp32[4] y = relu(x) output y }\n")
    assert "duplicate pipeline name" in str(caught.value)


# ---------------------------------------------------------------------------
# P1-13: the reference oracle must be total on the domain the device allows.
def test_reference_round_passes_nan_and_infinity_through():
    pipeline = _compile("pipeline p { input x : fp32[4] r = round(x) output r }")
    produced = _reference_run(pipeline, [float("nan"), float("inf"), 2.5, 3.5])["r"]
    assert math.isnan(produced[0])
    assert produced[1] == float("inf")
    assert produced[2] == 2.0 and produced[3] == 4.0


# ---------------------------------------------------------------------------
# P1-10: --emit json is documented as the machine-readable channel.
def test_emit_json_handles_multi_name_assignments(tmp_path):
    import contextlib
    import io
    import json

    from compiler.cli import main

    source = tmp_path / "multi.fgpu"
    source.write_text(
        "pipeline p { input x : fp32[4]\n"
        "             a, b = cumsum_totals(x)\n"
        "             output a\n"
        "             output b }\n")
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = main([str(source), "--emit", "json"])
    assert code == 0
    payload = json.loads(stream.getvalue())
    names = [value["name"] for value in payload["pipelines"][0]["values"]]
    assert names == ["a", "b"]


# ---------------------------------------------------------------------------
# P1-4: `HD = H * D` is linear_decode's derivation, not a magic first line.
def test_hd_dimension_is_not_shadowed_when_it_is_a_declared_input_dimension():
    pipeline = _compile(
        "pipeline p { input x : fp32[HD] y = relu(x) output y }")
    assert "HD = H * D" not in pipeline.triton.host
    decode = _compile(open("examples/batch8/linear_decode_forward.fgpu").read())
    assert "HD = H * D" in decode.triton.host


# ---------------------------------------------------------------------------
# P1-5: one namespace -- a name may not be both a value and a dimension, and
# an input may not shadow the host module it is launched with.
def test_a_computed_value_cannot_share_a_name_with_a_dimension_symbol():
    with pytest.raises(CompileError, match="ipeline 'p': the name 'D' is used both as a value and as a dimension symbol; rename one of them") as caught:
        compile_source("pipeline p { input x : fp32[N,D] D = softmax(x) output D }")
    assert "dimension symbol" in str(caught.value)


def test_an_input_may_share_a_name_with_a_dimension_because_hosts_rename_it():
    """The custom-op contract allows `input x : bf16[x,D]`; both hosts rename
    the parameter instead of emitting `def run_p(x, x)`."""
    pipeline = _compile(
        "pipeline collision { input x : bf16[x,D] y = relu(x) output y }")
    assert pipeline.triton.host.splitlines()[0].startswith("def run_collision(x_")
    reference_signature = pipeline.reference.host.splitlines()[0]
    assert reference_signature.startswith("def run_collision(x_")
    assert reference_signature.count("x") >= 1
    assert len({part.split("=")[0].strip() for part in
                reference_signature.split("(")[1].rstrip("):").split(",")}) == len(
        reference_signature.split("(")[1].rstrip("):").split(","))


def test_an_input_named_torch_does_not_shadow_the_launch_module():
    pipeline = _compile(
        "pipeline p { input torch : fp32[4] y = relu(torch) output y }")
    assert not pipeline.triton.host.startswith("def run_p(torch)")
    assert pipeline.triton.host.splitlines()[0].startswith("def run_p(torch_")


# ---------------------------------------------------------------------------
# P1-6: operator misuse is a diagnostic, not a Python exception.
def test_multi_result_misuse_is_diagnosable():
    cases = {
        "an integer literal only names": "pipeline p { input x : fp32[4] a, b = split(3) output a }",
        "produces several values": "pipeline p { input x : fp32[4] input y : fp32[4] z = add(split(x), y) output z }",
        "defines 2 results": "pipeline p { input x : fp32[4] a, b, c = cumsum_totals(x) output a }",
        "dtype literal cannot be an argument": "pipeline p { input x : fp32[4] y = cast_to(bf16, x) output y }",
        "shape literal cannot be an argument": "pipeline p { input x : fp32[4] y = reshape([2,2], x) output y }",
    }
    for fragment, source in cases.items():
        with pytest.raises(CompileError, match='error in pipeline') as caught:
            compile_source(source)
        assert fragment in str(caught.value), (fragment, str(caught.value))


# ---------------------------------------------------------------------------
# P1-6: a partially dead multi-result group still compiles in both backends.
def test_topk_last_keeps_a_single_live_result():
    pipeline = _compile(
        "pipeline p { input x : fp32[4]\n"
        "             values, ids = topk_last(x, 2)\n"
        "             output values }\n")
    assert len(pipeline.triton.kernels) == 1
    produced = _reference_run(pipeline, [1.0, 3.0, 2.0, 4.0])["values"]
    assert produced == [4.0, 3.0]


def test_segment_rank_keeps_a_single_live_result():
    pipeline = _compile(
        "pipeline p { input ids : int32[4]\n"
        "             counts, positions = segment_rank(ids, 2, 4)\n"
        "             output counts }\n")
    assert len(pipeline.triton.kernels) == 2      # reset is a no-op, rank counts
    produced = _reference_run(pipeline, [0, 1, 0, 1])["counts"]
    assert produced == [2, 2]


# ---------------------------------------------------------------------------
# P1-3: a literal-shape store_cache_4d host must not reference `PS`.
def test_store_cache_literal_shape_host_has_no_undefined_ps():
    pipeline = _compile(
        "pipeline sc {\n"
        " input k_view : fp32[2,2,1,2]\n"
        " input v_view : fp32[2,2,1,2]\n"
        " input cache_k : fp32[4,1,2]\n"
        " input cache_v : fp32[4,1,2]\n"
        " input loc : int64[4]\n"
        " k_out, v_out = store_cache_4d(k_view, v_view, cache_k, cache_v, loc, 2)\n"
        " output k_out\n"
        " output v_out }\n")
    host = pipeline.triton.host
    guard = [line for line in host.splitlines() if "page_size" in line]
    assert not any(" PS" in line or "PS:" in line for line in guard), guard
    assert any("k_view.shape[1]" in line for line in guard), guard
    assert "    PS = " not in host


# ---------------------------------------------------------------------------
# P1-9: `tl.arange` needs a power of two; a literal state extent need not be one.
def test_gated_delta_recurrence_widens_non_power_of_two_state_extents():
    pipeline = _compile(
        "pipeline gdn {\n"
        " input q:fp32[1,4,2,6]\n"
        " input k:fp32[1,4,2,6]\n"
        " input v:fp32[1,4,2,5]\n"
        " input g:fp32[1,4,2]\n"
        " input beta:fp32[1,4,2]\n"
        " input state0:fp32[1,2,5,6]\n"
        " o, s = gated_delta_recurrence(q, k, v, g, beta, state0)\n"
        " output o\n"
        " output s }\n")
    kernel = pipeline.triton.kernels[0]
    assert "KW: tl.constexpr" in kernel.signature
    assert "VW: tl.constexpr" in kernel.signature
    source = kernel.source
    assert "tl.arange(0, KW)" in source
    assert "tl.arange(0, VW)" in source
    assert source.count("mask=") >= 6      # every load and store is masked


# ---------------------------------------------------------------------------
# P1-11: the two reference hosts that used to raise a traceback now run, and a
# genuinely broken host would still be reported as a diagnostic, not a traceback.
def test_the_batch8_cache_examples_run_through_the_reference_host(capsys):
    from compiler.cli import main

    for path in ("examples/batch8/linear_decode_forward.fgpu",
                 "examples/batch8/store_cache_4d.fgpu"):
        code = main([path, "--run"])
        assert code == 0, (path, capsys.readouterr().err)


# ---------------------------------------------------------------------------
# P1-8: a kernel whose offsets are 32-bit arithmetic must not address a tensor
# that can span more than 2**31 - 1 addresses.
def test_the_shared_address_guard_reads_every_inputs_runtime_stride():
    pipeline = _compile(
        "pipeline p { input x : fp32[M,N] input y : fp32[M,N]\n"
        "             z = add(x, y) output z }")
    host = pipeline.triton.host
    branches = [line for line in host.splitlines() if "2147483647" in line]
    assert len(branches) == 1                  # one shared `if`
    assert " or " in branches[0]              # both inputs share it
    assert "x.stride(0)" in branches[0] and "y.stride(0)" in branches[0]
    assert "abs(" in branches[0]              # views may carry negative strides


def test_a_small_literal_input_still_gets_the_runtime_guard():
    # a literal declaration can still be a huge strided view at run time, so the
    # guard is about the runtime tensor, not the declared shape
    pipeline = _compile(
        "pipeline p { input x : fp32[4,5] y = relu(x) output y }")
    host = pipeline.triton.host
    assert "2**31 - 1 addresses" in host
    assert "x.stride(1)" in host


# ---------------------------------------------------------------------------
# P1-7: a 64-bit data index must not be truncated to 32 bits.
def test_gather_last_keeps_a_64_bit_index_wide():
    pipeline = _compile(
        "pipeline p { input x : fp32[8] input idx : int64[4]\n"
        "             y = gather_last(x, idx) output y }\n")
    source = pipeline.triton.kernels[0].source
    assert ".to(tl.int64)" in source
    assert ".to(tl.int32)" not in source
    # the reference contract: an out-of-bounds index reads zero
    produced = _reference_run(pipeline, [float(v) for v in range(8)],
                              [2 ** 32 + 3, 1, 7, -1])["y"]
    # 2**32 + 3 is out of bounds -> 0; -1 is normalised to the last element.
    assert produced == [0.0, 1.0, 7.0, 7.0]


def test_scatter_by_index_keeps_a_64_bit_index_wide():
    pipeline = _compile(
        "pipeline p { input base : fp32[8,4] input idx : int64[3] "
        "input values : fp32[3,4]\n"
        "             out = scatter_by_index(base, idx, values) output out }\n")
    source = "".join(kernel.source for kernel in pipeline.triton.kernels)
    assert ".to(tl.int64)" in source
    assert ".to(tl.int32)" not in source


# ---------------------------------------------------------------------------
# P1-2 / P1-9: linear_decode_forward's host and tile contract.
def test_linear_decode_caches_own_buffer_and_binds_its_block_literal():
    pipeline = _compile(
        "pipeline ld { input q:fp16[2,2,1,3] input k:fp16[2,2,1,3] "
        "input v:fp16[2,2,1,3] input cache:fp16[4,2,3,3] input slope:fp32[2] "
        "input slots:int32[2]\n"
        "             out, cache_out = linear_decode_forward("
        "q, k, v, cache, slope, slots, 8)\n"
        "             output out\n"
        "             output cache_out }\n")
    host = pipeline.triton.host
    # the cache output is a result with its own buffer, not the input tensor
    assert "cache.clone()" in host
    assert "cache, slot1" not in host.replace("cache, slope", "")
    assert "(q, k, v, cache, slope, slots, slot0, slot1," in host
    # BLOCK_SIZE is the declaration's literal, not a caller parameter
    assert host.splitlines()[0] == "def ld(q, k, v, cache, slope, slots):"
    assert "triton.cdiv(q.shape[3], 8)" in host
    # D is a runtime constexpr, so its lane tile is widened for tl.arange
    kernel = pipeline.triton.kernels[0]
    assert "DW: tl.constexpr" in kernel.signature
    assert "kd = tl.arange(0, DW)" in kernel.source
    assert "qm = kd < D" in kernel.source
    # and a non-power-of-two BLOCK_SIZE is a diagnostic, not a device failure
    with pytest.raises(CompileError, match='olumn 31: linear_decode_forward BLOCK_SIZE must be a power of two') as caught:
        compile_source(
            "pipeline ld { input q:fp16[2,2,1,3] input k:fp16[2,2,1,3] "
            "input v:fp16[2,2,1,3] input cache:fp16[4,2,3,3] input slope:fp32[2] "
            "input slots:int32[2]\n"
            "             out, cache_out = linear_decode_forward("
            "q, k, v, cache, slope, slots, 3)\n"
            "             output out\n"
            "             output cache_out }\n")
    assert "power of two" in str(caught.value)
