"""Executable fixtures for the measured PlatformConfig contracts.

The registry in ``test/platform_contract_registry.py`` names one owner per
PlatformConfig field.  This module supplies the fixtures for the contracts whose
consumer is emission or execution, and checks the registry's completeness:

* a new PlatformConfig field cannot land without being classified;
* every ``measured`` contract resolves to a real test (this module or another);
* the grid cap, reduction chunking, sigmoid spelling, GEMM tile and masked-tail
  contracts execute the emitted source through the CPU Triton double.

Device truth lives in the evidence documents cited by each registry entry; these
fixtures pin the compiler behaviour that those measurements selected.
"""
from __future__ import annotations

import math

import pytest

from compiler import compile_source
from compiler.backend.contracts import violations
from compiler.backend.platform_configs import PlatformConfig, resolve_platform
from compiler.backend.triton_backend import module
from test import platform_contract_registry as registry
from test import segmented_support


def _execute_emitted(code, inputs, pipeline):
    """CPU-double execution that keeps the emitted module-level constants.

    ``segmented_support.execute`` intentionally strips everything but functions;
    the platform fixtures need the ``BLOCK = ...`` constants the host reads, so
    this loader keeps top-level assignments as well.
    """
    import ast
    import math
    import types

    tl = segmented_support.TL()
    triton = segmented_support.Triton(tl)
    allocations = []

    def empty(shape, *, dtype, device):
        if isinstance(shape, int):
            shape = (shape,)
        buffer = segmented_support.Buffer([None] * math.prod(shape), shape, dtype)
        allocations.append(buffer)
        return buffer

    torch = types.SimpleNamespace(
        empty=empty, empty_like=lambda x: empty(x.shape, dtype=x.dtype, device=x.device),
        strided="strided", float16="fp16", float32="fp32", bfloat16="bf16",
        int32="int32", int64="int64", bool="bool")
    tree = ast.parse(code)
    tree.body = [node for node in tree.body
                 if (isinstance(node, ast.FunctionDef) and node.name != "_flowgpu_device_key")
                 or isinstance(node, ast.Assign)]
    namespace = {"tl": tl, "triton": triton, "torch": torch}
    exec(compile(ast.fix_missing_locations(tree), "<emitted-cpu-double>", "exec"), namespace)
    result = namespace["run_" + pipeline](*inputs)
    for buffer in allocations:
        assert len(buffer.writes) == len(buffer.data), ("unwritten CPU allocation", buffer.shape)
    return {key: buffer.data for key, buffer in result.items()}, triton.launches


def direct(name, **overrides):
    return resolve_platform(name, mapping_strategy="direct_grid", **overrides)


def _module(source, config, pipeline=0):
    program = compile_source(source, platform_config=config).pipelines[pipeline].triton
    assert program.complete, [str(m) for m in program.messages()]
    return program, module([program])


class _Device:
    type = "npu"


_NPU_DEVICE = _Device()


def _buffer(values, shape, dtype="fp16", *, npu=False):
    buffer = segmented_support.Buffer(list(values), shape, dtype)
    if npu:
        # The Ascend host asserts `a.device.type == 'npu'`; the CPU double's
        # default device is a plain string, so give it the attribute it reads.
        buffer.device = _NPU_DEVICE
    return buffer


# --- registry completeness -------------------------------------------------

def test_every_field_is_registered():
    import dataclasses

    fields = {field.name for field in dataclasses.fields(PlatformConfig)}
    registered = set(registry.CONTRACTS)
    assert registered - fields == set(), f"registry names unknown fields: {sorted(registered - fields)}"
    assert fields - registered == set(), f"unclassified PlatformConfig fields: {sorted(fields - registered)}"


def test_registry_kinds_are_known():
    assert {contract.kind for contract in registry.CONTRACTS.values()} <= {
        "measured", "policy", "plumbing", "identity"}


def test_every_measured_contract_has_a_real_test():
    missing = []
    for contract in registry.CONTRACTS.values():
        if contract.kind != "measured":
            continue
        path, test_name = contract.covered_by
        if path.startswith("fixture"):
            continue
        file = registry.__file__.rsplit("/", 1)[0] + "/../" + path
        import pathlib
        target = pathlib.Path(file).resolve()
        if not target.exists():
            missing.append((contract.field, path, "file missing"))
        elif test_name and f"def {test_name}(" not in target.read_text(errors="ignore"):
            missing.append((contract.field, path, f"no test {test_name}"))
    assert not missing, missing


def test_registry_points_at_this_module_for_inline_fixtures():
    import pathlib

    here = pathlib.Path(__file__).read_text()
    for contract in registry.CONTRACTS.values():
        path, name = contract.covered_by
        if path == "fixture":
            assert f"def {name}(" in here, f"fixture {name} for {contract.field} is missing"


# --- executable fixtures ---------------------------------------------------

ADD_1000 = "pipeline p { input a:fp16[1000] input b:fp16[1000] out = add(a, b) output out }"
ADD_1024 = "pipeline p { input a:fp16[1024] input b:fp16[1024] out = add(a, b) output out }"
SIGMOID_256 = "pipeline p { input x:fp16[256] out = sigmoid(x) output out }"
SUM_4_2048 = "pipeline p { input x:fp16[4,2048] out = sum_last(x) output out }"
MATMUL_64 = "pipeline m { input a:fp16[64,64] input b:fp16[64,64] out = matmul(a, b) output out }"


def test_masked_load_other_contract_is_audited_and_executed():
    """A tail-masked load with `other=` must execute, and the audit must accept
    it on every target (the measured defect is a *broader store mask*)."""
    for name in ("cuda", "ascend", "ascend-910b", "kunlunxin"):
        npu = name.startswith("ascend")
        a = _buffer([0.25] * 1000, (1000,), npu=npu)
        b = _buffer([1.5] * 1000, (1000,), npu=npu)
        config = direct(name)
        _, text = _module(ADD_1000, config)
        problems = violations(text, config)
        assert problems == [], (name, problems)
        out, _ = _execute_emitted(text, (a, b), pipeline="p")
        got = out["out"]
        assert all(abs(value - 1.75) < 1e-3 for value in got)
        # the CPU double asserts every allocation was fully written


def test_max_grid_dim_caps_the_launch_and_writes_every_element():
    a = _buffer([1.0] * 1024, (1024,))
    b = _buffer([2.0] * 1024, (1024,))
    config = resolve_platform("cuda", mapping_strategy="grid_stride_loop",
                              device_cores=8, max_grid_dim=3, block_size=64)
    _, text = _module(ADD_1024, config)
    out, launches = _execute_emitted(text, (a, b), pipeline="p")
    assert all(abs(value - 3.0) < 1e-3 for value in out["out"])
    assert all(grid[0] <= 3 for _, grid in launches), launches


def test_reduction_block_limit_chunks_and_stays_correct():
    rows, columns = 4, 2048
    values = [1.0] * (rows * columns)
    buffer = _buffer(values, (rows, columns))
    config = resolve_platform("cuda", reduction_block_limit=256)
    _, text = _module(SUM_4_2048, config)
    assert "for " in text  # a chunked sweep, not one huge lane range
    out, _ = _execute_emitted(text, (buffer,), pipeline="p")
    assert all(abs(value - columns) < 1e-1 for value in out["out"])


def test_sigmoid_impl_switches_emission_and_executes():
    native_text = _module(SIGMOID_256, resolve_platform("cuda"))[1]
    exp2_text = _module(SIGMOID_256, resolve_platform("cuda", sigmoid_impl="exp2"))[1]
    assert "tl.sigmoid" in native_text
    assert "tl.sigmoid" not in exp2_text and "tl.exp2" in exp2_text
    values = [-2.0, -0.5, 0.0, 0.5, 2.0] * 51 + [1.0]
    buffer = _buffer(values, (256,))
    out, _ = _execute_emitted(exp2_text, (buffer,), pipeline="p")
    expected = [1.0 / (1.0 + math.exp(-value)) for value in values]
    assert max(abs(a - b) for a, b in zip(out["out"], expected)) < 1e-2


def test_matmul_tile_policy_threads_header_and_launch():
    config = resolve_platform("cuda", matmul_block_m=32, matmul_block_n=64,
                              matmul_block_k=16, matmul_num_warps=8,
                              matmul_num_stages=4)
    _, text = _module(MATMUL_64, config)
    assert "BLOCK_M = 32" in text and "BLOCK_N = 64" in text and "BLOCK_K = 16" in text
    assert "num_warps=8" in text and "num_stages=4" in text
    default_text = _module(MATMUL_64, resolve_platform("cuda"))[1]
    # default = measured 2026-10-27 floor (128x128x64, 3 stages, 8 warps)
    assert "BLOCK_M = 128" in default_text and "BLOCK_K = 64" in default_text
    assert "num_warps=8" in default_text and "num_stages=3" in default_text
    legacy = resolve_platform("cuda", matmul_block_m=64, matmul_block_n=64,
                              matmul_block_k=32, matmul_num_warps=4,
                              matmul_num_stages=2)
    legacy_text = _module(MATMUL_64, legacy)[1]
    assert "BLOCK_M = 64" in legacy_text and "num_stages=2" in legacy_text


def test_cuda_default_gemm_tile_is_the_measured_floor():
    """Product default: the measured 128x128x64/3-stage/8-warp tile, not the
    historical 64x64x32.  An explicit platform-wide launch knob still wins."""
    default = resolve_platform("cuda")
    assert (default.matmul_block_m, default.matmul_block_n, default.matmul_block_k) == (128, 128, 64)
    assert (default.matmul_num_warps, default.matmul_num_stages) == (8, 3)
    overridden = resolve_platform("cuda", num_warps=4, num_stages=2)
    assert overridden.matmul_num_warps is None and overridden.num_warps == 4
    assert overridden.matmul_num_stages is None and overridden.num_stages == 2
    pinned = resolve_platform("cuda", matmul_block_m=64, matmul_num_warps=4)
    assert pinned.matmul_block_m == 64 and pinned.matmul_num_warps == 4


def test_group_m_swizzle_is_emitted():
    _, text = _module(MATMUL_64, resolve_platform("cuda", group_m=8))
    assert "GROUP_M" in text and "pid_in_group" in text
    default_text = _module(MATMUL_64, resolve_platform("cuda"))[1]
    assert "GROUP_M" not in default_text


def test_2d_tile_policy_changes_emission():
    """The Ascend profile records the measured 2-D tile; the consumer is the
    indexed lowering, so this pins the recorded values and their validation."""
    ascend = direct("ascend")
    assert ascend.vector_tile_2d is True
    assert (ascend.tile_rows, ascend.tile_cols) == (2, 512)
    cuda = direct("cuda")
    assert cuda.vector_tile_2d is False and cuda.tile_rows == cuda.tile_cols == 1
    with pytest.raises(ValueError, match="tile_rows"):
        PlatformConfig("probe", tile_rows=3)
    with pytest.raises(ValueError, match="tile_cols"):
        PlatformConfig("probe", tile_cols=0)


def test_block_ptr_contract_is_enforced_per_target():
    source = """
@triton.jit
def k(x_ptr, out_ptr, N: tl.constexpr):
    b = tl.make_block_ptr(base=x_ptr, shape=(N,), strides=(1,), offsets=(0,), block_shape=(64,), order=(0,))
    v = tl.load(b, boundary_check=(0,), padding_option="zero")
    tl.store(out_ptr + tl.arange(0, 64), v)
"""
    assert violations(source, direct("cuda")) == []
    assert violations(source, direct("ascend")), "Ascend must flag the block_ptr path"


def test_gather_and_loop_store_contracts_are_audited():
    gather = """
@triton.jit
def k(idx_ptr, img_ptr, out_ptr, TOTAL):
    e = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    which = tl.load(idx_ptr + e, mask=e < TOTAL, other=0)
    v = tl.load(img_ptr + which, mask=e < TOTAL, other=0.0)
    tl.store(out_ptr + e, v, mask=e < TOTAL)
"""
    loop_store = """
@triton.jit
def k(src_ptr, out_ptr, n):
    off = tl.arange(0, 64)
    acc = tl.zeros((64,), tl.float32)
    for k in range(0, n):
        acc += tl.load(src_ptr + k * 64 + off)
        tl.store(out_ptr + k * 64 + off, acc)
"""
    for source, token in ((gather, "gather"), (loop_store, "loop")):
        problems = violations(source, direct("kunlunxin"))
        assert problems and token in problems[0]
        for name in ("cuda", "ascend", "ascend-910b"):
            assert violations(source, direct(name)) == [], (name, source[:20])
