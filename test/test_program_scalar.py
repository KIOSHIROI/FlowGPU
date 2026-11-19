"""Program-uniform scalar + the affine-write contract.

Two capabilities, one measurement. The hand-written Task153 kernel is
program-uniform : one scalar load per program (`cu_seqlens[seq]`,
`chunk_indices[c,0]`) and then every address is
`scalar_base + affine(arange)`. The two programs that collapsed on 昆仑芯
(Task152 0.03x, Task154 0.02-0.05x) were the ones that could not be written
that way. So this file pins:

* the IR value category -- `ProgramScalar`: one integer load per program, from
  a rank-1 integer table, index scalar/program-id-derived, fail-closed bounds;
* its emission -- one unmasked scalar load at the top of the kernel body (not
  re-derived per lane), folded into a base, with a host bounds guard;
* the static check -- `analysis/write_address` classifies every ``tl.store``
  address of emitted source as affine / lane-division-modulo / data-dependent
  and the affine-write targets refuse the latter before a kernel is released.

The numeric oracle executes the **emitted source** on the CPU Triton double in
``test/segmented_support.py`` with a real grid, so "one value per program" is
tested as a value (the oracle is `table[i // BLOCK]` over the output index), not
as a text assertion. Everything here is CPU evidence; the board is the device
experiment.

The user-visible operator is deliberately **not** registered in
``frontend/operator.py``: the tests register the staged rule through
``frontend/program_scalar.register()`` and remove it again, so every default
build in the suite is byte-identical and the operator step can decide the
spelling later.
"""
import ast
import math
import types

import pytest

from flowgpu.compiler.analysis import write_address
from flowgpu.compiler.backend.platform_configs import resolve_platform
from flowgpu.compiler.backend.triton_backend import module
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.frontend.diagnostic import CompileError
from flowgpu.compiler.frontend.program_scalar import (
    STAGED_OPERATOR,
    infer_program_scalar,
    register as register_program_scalar,
    unregister as unregister_program_scalar,
)
from flowgpu.compiler.ir.program_scalar_ir import (
    PROGRAM_SCALAR_INDEX,
    STAGED_PROGRAM_SCALAR_OP,
    AddressTerm,
    AffineAddress,
    LaneIndex,
    ProgramIndex,
    ProgramScalar,
    ScalarRef,
)
from test import segmented_support
from flowgpu.compiler.frontend.diagnostic import RuleError

TEXT256 = (
    "pipeline p { input table:int32[S] input x:int32[M,N] "
    "b=program_scalar(table) y=add(x,b) output y }"
)
# The same value consumed by a value-layer primitive, which routes the region
# through the indexed emitter instead of the flat pointwise one.
INDEXED = (
    "pipeline p { input table:int32[S] input x:int32[M,N] "
    "b=program_scalar(table) u=add(x,b) y=maximum(u,x) output y }"
)
BLOCK = 1024


@pytest.fixture
def staged_operator():
    register_program_scalar()
    try:
        yield
    finally:
        unregister_program_scalar()


# ---------------------------------------------------------------------------
# the IR contract: what the value category is, and what it refuses
# ---------------------------------------------------------------------------


def test_the_staged_frontend_name_is_the_ir_name():
    assert STAGED_OPERATOR == STAGED_PROGRAM_SCALAR_OP


def test_the_staged_operator_is_not_in_the_default_registry():
    from flowgpu.compiler.frontend.operator import OPERATORS
    assert STAGED_OPERATOR not in OPERATORS


def test_program_scalar_reads_one_integer_scalar_from_a_rank_one_table():
    spec = ProgramScalar(table="cu_seqlens", index=ProgramIndex(axis=0))
    assert spec.program_uniform
    assert spec.host_checkable
    assert spec.as_dict()["table"] == "cu_seqlens"
    assert spec.as_dict()["index"] == dict(kind="program", axis=0, scale=1,
                                           offset=0)


@pytest.mark.parametrize("dtype", ["fp32", "bf16", "bool"])
def test_a_program_scalar_cannot_declare_a_non_integer_dtype(dtype):
    with pytest.raises(ValueError, match="program scalar dtype must be int32 or int64, got '") as caught:
        ProgramScalar(table="t", index=ProgramIndex(), dtype=dtype)
    assert "int32 or int64" in str(caught.value)


def test_a_chained_device_index_needs_an_explicit_trusted_contract():
    # `cu_seqlens[seq]` where `seq` is itself a program-uniform scalar cannot be
    # bounded by the host. Fail-closed by default; the caller may accept.
    with pytest.raises(ValueError, match="program scalar index 'seq' is itself a device value, so the host cannot prove it is in bounds; pass bounds='trusted' to accept the caller's contract") as caught:
        ProgramScalar(table="cu_seqlens",
                      index=ScalarRef("seq", host_visible=False))
    assert "bounds='trusted'" in str(caught.value)
    trusted = ProgramScalar(table="cu_seqlens",
                            index=ScalarRef("seq", host_visible=False),
                            bounds="trusted")
    assert not trusted.index.host_visible


def test_program_scalar_rejects_an_unknown_bounds_policy_and_negative_literal():
    with pytest.raises(ValueError, match="program scalar bounds must be reject or trusted, got 'mask'"):
        ProgramScalar(table="t", index=ProgramIndex(), bounds="mask")
    with pytest.raises(ValueError, match='a literal program scalar index must be non-negative'):
        ProgramScalar(table="t", index=-1)


def test_affine_address_admits_a_data_base_but_not_a_data_coefficient():
    address = AffineAddress(
        label="gcu", base=ProgramScalar(table="cu", index=ProgramIndex()),
        terms=(AddressTerm(coefficient=1, index=LaneIndex("lane")),),
    )
    assert "cu_scalar" in address.expression()
    # A data-derived *coefficient* is not an affine term: it would make the
    # lane scale data-dependent, which is the shape the targets collapse on.
    with pytest.raises(ValueError, match='address term coefficient must be an int or an identifier') as caught:
        AddressTerm(coefficient=ScalarRef("scale"), index=LaneIndex("lane"))
    assert "coefficient" in str(caught.value)


def test_affine_address_terms_are_lane_or_program_uniform_only():
    with pytest.raises(ValueError, match='address term index must be a LaneIndex or ProgramIndex; a data-derived coefficient or a scalar scale factor is not an affine address term'):
        AddressTerm(coefficient=1, index=ScalarRef("anything"))


# ---------------------------------------------------------------------------
# type checking: the staged rule
# ---------------------------------------------------------------------------


def test_the_staged_rule_returns_a_rank_zero_integer_value():
    from flowgpu.compiler.frontend.ast_node import TensorType
    result = infer_program_scalar([TensorType("int32", ["S"])])
    assert result.is_scalar
    assert result.dtype == "int32"


@pytest.mark.parametrize("table", [
    ("fp32", ["S"]),          # floating index
    ("int32", ["M", "N"]),    # table with more than one axis
])
def test_the_staged_rule_refuses_a_non_scalar_read(table, staged_operator):
    from flowgpu.compiler.frontend.ast_node import TensorType
    dtype, shape = table
    with pytest.raises(RuleError, match='program_scalar'):
        infer_program_scalar([TensorType(dtype, shape)])
    with pytest.raises(CompileError, match='pe error in pipeline'):
        compile_source(
            "pipeline p { input x:fp32[M,N] input t:"
            + dtype + "[" + ",".join(shape) + "] "
            "b=program_scalar(t) y=add(x,b) output y }"
        )


def test_a_program_scalar_is_fused_into_the_kernel_that_consumes_it(staged_operator):
    program = compile_source(TEXT256).pipelines[0]
    kernel = program.triton.kernels[0]
    # One kernel, one operator chain: the producer is a register, not a buffer.
    assert len(program.triton.kernels) == 1
    assert kernel.ops == ("program_scalar", "add")
    assert "b fused by 13.6" in kernel.source


# ---------------------------------------------------------------------------
# emission: one scalar load per program, folded into the expression
# ---------------------------------------------------------------------------


def test_emission_hoists_one_unmasked_uniform_load(staged_operator):
    kernel = compile_source(TEXT256).pipelines[0].triton.kernels[0]
    source = kernel.source

    assert source.count("tl.load(") == 2          # the scalar + the tile
    scalar_line = next(line for line in source.splitlines()
                       if "= tl.load(table_ptr" in line)
    # Program-indexed address: no `tl.arange` in it, so the load is uniform...
    assert "tl.arange" not in scalar_line
    assert "mask=" not in scalar_line
    # ...and the table's declared integer dtype is pinned (the staged spec used
    # to default to int64 even when the table was int32; review P2).
    assert ".to(tl.int32)" in scalar_line
    # The value is used directly, not re-derived per lane.
    assert "+ b_scalar)" in source


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_the_scalar_load_keeps_the_tables_declared_dtype(dtype, staged_operator):
    source = (f"pipeline p {{ input table:{dtype}[S] input x:int32[M,N] "
              "b=program_scalar(table) y=add(x,b) output y }")
    kernel = compile_source(source).pipelines[0].triton.kernels[0]
    scalar_line = next(line for line in kernel.source.splitlines()
                       if "= tl.load(table_ptr" in line)
    assert f".to(tl.{dtype})" in scalar_line


def test_the_scalar_load_is_at_the_kernel_top_level(staged_operator):
    """One load per program, not inside any lane loop."""
    kernel = compile_source(TEXT256).pipelines[0].triton.kernels[0]
    lines = kernel.source.splitlines()
    scalar = next(line for line in lines if "= tl.load(table_ptr" in line)
    # Four spaces: the kernel's top level. A per-lane re-derivation would be
    # nested inside a loop or a broadcast expression.
    assert scalar.startswith("    ") and not scalar.startswith("        ")
    assert "for " not in kernel.source.split("tl.load(table_ptr")[0].splitlines()[-1]


def test_the_indexed_emitter_emits_the_same_uniform_load(staged_operator):
    program = compile_source(INDEXED).pipelines[0]
    kernel = program.triton.kernels[0]
    assert kernel.kind == "primitive"       # the value-layer path
    scalar_lines = [line for line in kernel.source.splitlines()
                    if "= tl.load(table_ptr" in line]
    assert len(scalar_lines) == 1
    assert "tl.arange" not in scalar_lines[0]
    assert "mask=" not in scalar_lines[0]
    assert "b_scalar" in kernel.source


def test_the_emitted_store_address_stays_affine(staged_operator):
    kernel = compile_source(TEXT256).pipelines[0].triton.kernels[0]
    addresses = write_address.store_addresses(kernel.source)
    assert addresses and all(address.affine for address in addresses)


def test_emission_records_the_bounds_the_host_must_enforce(staged_operator):
    kernel = compile_source(TEXT256).pipelines[0].triton.kernels[0]
    (bound,) = kernel.program_scalar_bounds
    assert bound["table"] == "table"
    assert bound["index"] == PROGRAM_SCALAR_INDEX[STAGED_OPERATOR]
    assert "triton.cdiv" in bound["grid_count"]


def test_the_host_guards_the_table_extent_before_launching(staged_operator):
    program = compile_source(TEXT256).pipelines[0]
    host = program.triton.host
    guard = [line for line in host.splitlines()
             if "program_scalar index for table table" in line]
    assert guard, host
    # The guard is emitted before the allocation/launch.
    assert host.index("program_scalar index") < host.index("torch.empty")


def test_the_reference_backend_refuses_and_names_the_cpu_double(staged_operator):
    program = compile_source(TEXT256).pipelines[0]
    assert not program.reference.complete
    message = program.reference.messages()[0]
    assert "launch mapping" in message
    assert "CPU Triton double" in message


# ---------------------------------------------------------------------------
# numeric oracle: execute the emitted source on the CPU Triton double
# ---------------------------------------------------------------------------


def execute_emitted(code, inputs, pipeline):
    """Exec an emitted module against the small CPU Triton double.

    Imports and the device-key helper are dropped (there is no torch/triton on
    this machine); every other module-level name is kept, because the host's
    grid expression uses the emitted `BLOCK` constant.
    """
    language = segmented_support.TL()
    triton = segmented_support.Triton(language)
    allocations = []

    def empty(shape, *, dtype, device):
        if isinstance(shape, int):
            shape = (shape,)
        buffer = segmented_support.Buffer([None] * math.prod(shape), shape, dtype)
        allocations.append(buffer)
        return buffer

    torch = types.SimpleNamespace(
        empty=empty,
        empty_like=lambda x: empty(x.shape, dtype=x.dtype, device=x.device),
        strided="strided", float16="fp16", bfloat16="bf16", float32="fp32",
        int32="int32", int64="int64", bool="bool",
    )
    tree = ast.parse(code)
    tree.body = [
        node for node in tree.body
        if not isinstance(node, (ast.Import, ast.ImportFrom))
        and not (isinstance(node, ast.FunctionDef)
                 and node.name == "_flowgpu_device_key")
    ]
    namespace = {"tl": language, "triton": triton, "torch": torch}
    exec(compile(ast.fix_missing_locations(tree), "<emitted-cpu-double>", "exec"),
         namespace)
    result = namespace["run_" + pipeline](*inputs)
    for buffer in allocations:
        assert len(buffer.writes) == len(buffer.data), ("unwritten", buffer.shape)
    return {name: buffer.data for name, buffer in result.items()}, triton


def _buffers(rows, columns, table_size):
    table = segmented_support.Buffer(list(range(table_size)), (table_size,), "int32")
    x = segmented_support.Buffer([(index % 53) - 20 for index in range(rows * columns)],
                                 (rows, columns), "int32")
    return table, x


@pytest.mark.parametrize("source", [TEXT256, INDEXED])
def test_one_value_per_program_matches_an_independent_oracle(source, staged_operator):
    rows, columns, table_size = 64, 32, 8       # numel 2048 -> 2 programs
    program = compile_source(source).pipelines[0].triton
    values, triton = execute_emitted(module([program]),
                                     _buffers(rows, columns, table_size), "p")
    expected = [
        ((index % 53) - 20) + (index // BLOCK)
        for index in range(rows * columns)
    ]
    assert triton.launches == [("p_k0", (2,))]
    assert values["y"] == expected


def test_the_host_refuses_an_out_of_bounds_table_before_any_launch(staged_operator):
    rows, columns = 64, 24
    program = compile_source(TEXT256).pipelines[0].triton
    code = module([program])
    table, x = _buffers(rows, columns, table_size=1)   # 1536 elements -> 2 programs
    # One program per 1024 elements, so the launch can reach index 1 while the
    # table has one entry.
    with pytest.raises(ValueError, match='program_scalar index for table table is out of bounds: the launch can reach 1 but the table has 1 entries') as caught:
        execute_emitted(code, [table, x], "p")
    assert "program_scalar index for table table is out of bounds" in str(caught.value)


def test_a_table_long_enough_for_the_grid_is_accepted(staged_operator):
    rows, columns = 64, 24
    program = compile_source(TEXT256).pipelines[0].triton
    table, x = _buffers(rows, columns, table_size=2)
    values, _ = execute_emitted(module([program]), [table, x], "p")
    assert len(values["y"]) == rows * columns


# ---------------------------------------------------------------------------
# the write-address contract
# ---------------------------------------------------------------------------


def addresses(source):
    return write_address.store_addresses(source)


def test_flat_pointwise_store_is_affine():
    source = """def k(x_ptr, out_ptr, numel, BLOCK: tl.constexpr, x_contiguous: tl.constexpr, x_dim0: tl.constexpr, x_stride0: tl.constexpr, x_dim1: tl.constexpr, x_stride1: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    a0 = offsets
    if not x_contiguous:
        a0 = (((offsets) // (x_dim1))) * x_stride0 + (((offsets) // (1)) % x_dim1) * x_stride1
    t0 = tl.load(x_ptr + a0, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, t0, mask=mask)"""
    found = addresses(source)
    assert found and all(address.affine for address in found)


def test_two_dimensional_tile_store_is_affine():
    source = """def k(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    col = tl.program_id(1) * 64 + tl.arange(0, 64)
    mask = (row[:, None] < M) & (col[None, :] < N)
    t0 = tl.load(x_ptr + row[:, None] * N + col[None, :], mask=mask, other=0)
    tl.store(out_ptr + row[:, None] * N + col[None, :], t0, mask=mask)"""
    assert all(address.affine for address in addresses(source))


def test_program_id_division_is_uniform_and_still_affine():
    """`pid // TILES` is a scalar per program, not a per-lane division."""
    source = """def k(x_ptr, out_ptr, rows, width):
    row = tl.program_id(0) // (rows)
    col = (tl.program_id(0) % (rows)) * 128 + tl.arange(0, 128)
    tl.store(out_ptr + row * (width) + col, 0, mask=col < width)"""
    assert all(address.affine for address in addresses(source))


def test_a_program_uniform_scalar_base_is_affine():
    """The capability this task adds: data folded into a base, not per lane."""
    source = """def k(table_ptr, out_ptr, BLOCK: tl.constexpr):
    base = tl.load(table_ptr + tl.program_id(0))
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + base + offsets, 0)"""
    (address,) = addresses(source)
    assert address.affine
    assert address.uniform_load


@pytest.mark.parametrize("operator", ["//", "%"])
def test_per_lane_division_modulo_is_refused(operator):
    body = "shifted = idx % N" if operator == "%" else "shifted = idx // N"
    source = f"""def k(out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    {body}
    tl.store(out_ptr + shifted, 0)"""
    (address,) = addresses(source)
    assert address.verdict == write_address.LANE_DIVMOD
    message = write_address.violations(source, "k")[0]
    assert "`" + operator + "`" in message
    assert "lane term 'idx'" in message
    assert "line 4" in message
    assert "scalar_base + affine(arange)" in message


def test_a_per_lane_load_in_the_store_address_is_refused():
    source = """def k(table_ptr, out_ptr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    loc = tl.load(table_ptr + idx)
    tl.store(out_ptr + loc * 16 + idx, 0)"""
    (address,) = addresses(source)
    assert address.verdict == write_address.LANE_LOAD
    message = write_address.violations(source, "k")[0]
    assert "data-dependent per lane" in message
    assert "folded into the base" in message


def test_a_uniform_load_used_as_a_lane_coefficient_is_refused():
    source = """def k(table_ptr, out_ptr, BLOCK: tl.constexpr):
    scale = tl.load(table_ptr + tl.program_id(0))
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + scale * idx, 0)"""
    (address,) = addresses(source)
    assert address.verdict == write_address.LANE_COEFFICIENT
    assert "scalar base" in write_address.violations(source, "k")[0]


# ---------------------------------------------------------------------------
# platform policy and the pre-emission gate
# ---------------------------------------------------------------------------


def test_only_the_measured_targets_require_affine_write_addresses():
    assert resolve_platform("cuda").affine_write_addresses is False
    assert resolve_platform("kunlunxin").affine_write_addresses is True
    assert resolve_platform("ascend", device_cores=48).affine_write_addresses is True
    assert resolve_platform("ascend-910b",
                            device_cores=48).affine_write_addresses is True


def test_the_policy_can_be_overridden_explicitly():
    assert resolve_platform("cuda", affine_write_addresses=True).affine_write_addresses
    assert not resolve_platform("kunlunxin",
                                affine_write_addresses=False).affine_write_addresses


SCATTER = (
    "pipeline sc { input base:fp32[8,16] input index:int64[4] input values:fp32[4,16] "
    "out=scatter_by_index(base,index,values) output out }"
)


def test_a_default_cuda_build_is_untouched_by_the_gate():
    program = compile_source(SCATTER).pipelines[0]
    assert program.triton.complete
    assert len(program.triton.kernels) == 2
    flagged = write_address.audit((program.triton,))
    assert any(entry["verdict"] == write_address.LANE_LOAD for entry in flagged)


def test_an_affine_write_target_refuses_a_non_affine_store_before_emitting():
    config = resolve_platform("kunlunxin",
                              allowed_kernel_kinds=("pointwise", "scatter_by_index"))
    with pytest.raises(CompileError, match='\\); lane dependency: offset, target\\); a program-uniform scalar \\(one load per program\\) must be folded into the base; a per-lane load makes the store a scatter\\. This target sustains only scalar_base \\+ affine\\(arange\\) writes') as caught:
        compile_source(SCATTER, platform_config=config)
    message = str(caught.value)
    assert "affine write address required" in message
    assert "store address" in message
    assert "lane dependency" in message


def test_a_program_scalar_kernel_passes_the_affine_write_gate(staged_operator):
    config = resolve_platform("kunlunxin")
    program = compile_source(TEXT256, platform_config=config).pipelines[0]
    assert program.triton.complete
    assert all(address.affine
               for kernel in program.triton.kernels
               for address in write_address.store_addresses(kernel.source))


# ---------------------------------------------------------------------------
# evidence: what the check finds in the existing corpus
# ---------------------------------------------------------------------------

#: The two example kernels in this corpus whose stores are not affine, and why.
#: Everything else in `examples/` is a flat run, a tile or a row-wise affine
#: store -- which is what "the default corpus is unaffected" means, checked
#: here rather than asserted in prose.
NON_AFFINE_EXAMPLES = {
    "examples/batch8/store_cache_4d.fgpu": "cache_store",
    "examples/batch8/gather_spec_extras.fgpu": "primitive",
}


@pytest.mark.parametrize("path,kind", sorted(NON_AFFINE_EXAMPLES.items()))
def test_the_check_finds_the_known_non_affine_store(path, kind):
    from pathlib import Path
    program = compile_source(Path(path).read_text()).pipelines[0].triton
    kernels = {kernel.kind: kernel for kernel in program.kernels}
    assert kind in kernels
    verdicts = {address.verdict
                for address in write_address.store_addresses(kernels[kind].source)}
    assert verdicts & {write_address.LANE_DIVMOD, write_address.LANE_LOAD,
                       write_address.NON_AFFINE}


def test_the_pointwise_and_rowwise_corpus_has_only_affine_writes():
    from pathlib import Path
    for path in ("examples/residual_norm.fgpu",
                 "examples/batch8/wide_row_softmax.fgpu"):
        program = compile_source(Path(path).read_text()).pipelines[0].triton
        for kernel in program.kernels:
            for address in write_address.store_addresses(kernel.source):
                if kernel.kind in {"pointwise", "rowwise"}:
                    assert address.affine, (path, kernel.name, address)


def test_host_checkable_matches_what_the_host_guard_does():
    """The IR property is not decorative: the backend reads it."""
    from flowgpu.compiler.backend.program_scalar import host_guards
    from flowgpu.compiler.ir.program_scalar_ir import ScalarRef

    trusted_device = ProgramScalar(
        table="table", index=ScalarRef(name="idx", host_visible=False),
        bounds="trusted")
    assert trusted_device.host_checkable is False
    bound = dict(table="table", index=trusted_device.index, dtype="int32",
                 grid_count=4, host_checkable=trusted_device.host_checkable)
    assert host_guards([bound], {"table": "table_ptr"}) == []

    checkable = ProgramScalar(table="table", index=ProgramIndex(axis=0),
                              bounds="trusted")
    assert checkable.host_checkable is True
    bound = dict(table="table", index=checkable.index, dtype="int32",
                 grid_count=4, host_checkable=checkable.host_checkable)
    lines = host_guards([bound], {"table": "table_ptr"})
    assert any("table_ptr.shape[0]" in line for line in lines)
