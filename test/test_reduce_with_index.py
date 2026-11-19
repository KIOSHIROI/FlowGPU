"""`reduce_with_index`: the index-carrying reduction, `argmax_last`/`argmin_last`.

`argmax_last` and `argmin_last` are ONE structural language feature, not two:
they share one type rule, one lowering and one kernel kind. The tests are
grouped by the claim each holds: the registry/type rules, the contract the
lowering must obey on a measured target, the reference semantics against an
independent oracle, and the four semantic decisions the operator makes rather
than inherits (tie order, NaN, +-inf, and the all-NaN row).

The reference backend is one implementation and the `_oracle` below is a second,
written from the contract instead of from the lowering. A test comparing them is
a comparison, not a tautology.
"""
import ast
import math
from dataclasses import replace

import pytest

from flowgpu.compiler.analysis.write_address import store_addresses
from flowgpu.compiler.analysis.write_address import violations as address_violations
from flowgpu.compiler.backend.contracts import violations as contract_violations
from flowgpu.compiler.backend.platform_configs import PLATFORMS
from flowgpu.compiler.backend.reference import inputs_for, module as reference_module
from flowgpu.compiler.backend.reference import run as run_reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.frontend.ast_node import TensorType
from flowgpu.compiler.frontend.diagnostic import RuleError
from flowgpu.compiler.frontend.diagnostic import CompileError
from flowgpu.compiler.frontend.operator import (
    OPERATORS,
    infer_argmax_last,
    infer_argmin_last,
)

OPERATORS_UNDER_TEST = ("argmax_last", "argmin_last")


def shape_text(shape):
    return "[" + ",".join(str(extent) for extent in shape) + "]"


def program(dtype, shape, op, name="pick"):
    return f"""
pipeline {name} {{
 input x:{dtype}{shape_text(shape)}
 values, ids = {op}(x)
 output values
 output ids
}}
"""


def compile_program(source, **kwargs):
    return compile_source(source, **kwargs).pipelines[0]


def feeds(pipeline):
    return {name: values for name, _, _, values in inputs_for(pipeline.reference, {})}


def execute(pipeline, tensors):
    """Run the *reference lowering* on caller-supplied data.

    `run_reference` generates its own data, which has no ties or NaNs; the
    semantic decisions need hand-built rows, so this executes the same generated
    host with those rows.
    """
    namespace = {}
    exec(compile(reference_module([pipeline.reference]), "<reference>", "exec"),
         namespace)
    host = namespace[f"run_{pipeline.reference.pipeline}"]
    return host(*tensors)


def oracle(x, width, maximum):
    """The contract, rewritten from scratch: last-axis winner, lowest index.

    NaN is skipped (it is never a candidate); +-inf is an ordinary value. With
    no candidate at all the answer is the direction's identity and index 0.
    """
    rows = len(x) // width
    values, indices = [], []
    for row in range(rows):
        best = None
        for column in range(width):
            value = float(x[row * width + column])
            if value != value:
                continue
            if best is None or ((value > best[0]) if maximum else (value < best[0])) \
                    or (value == best[0] and column < best[1]):
                best = (value, column)
        if best is None:
            values.append(-math.inf if maximum else math.inf)
            indices.append(0)
        else:
            values.append(best[0])
            indices.append(best[1])
    return values, indices


def reduce_kernels(pipeline):
    return [kernel for kernel in pipeline.triton.kernels
            if kernel.kind == "reduce_with_index"]


# --------------------------------------------------------------------------
# one feature, two names
# --------------------------------------------------------------------------
def test_the_two_names_are_one_registered_feature():
    """The audit's denominator counts this as 1; the registry must show why."""
    for name in OPERATORS_UNDER_TEST:
        spec = OPERATORS[name]
        assert spec.arity == 1
        assert spec.multi is True
        assert spec.pointwise is False
        assert spec.acc is None      # selecting an extreme accumulates nothing

    from types import SimpleNamespace
    from flowgpu.compiler.backend.backends import KERNELS, kernel_kind

    assert set(KERNELS["reduce_with_index"]) == set(OPERATORS_UNDER_TEST)
    for name in OPERATORS_UNDER_TEST:
        assert kernel_kind(SimpleNamespace(op=name)) == "reduce_with_index"


def test_the_two_results_are_a_values_tensor_and_an_int32_index_tensor():
    pipeline = compile_program(program("fp32", [4, 8], "argmax_last"))
    values = pipeline.symbols.lookup("values").type
    ids = pipeline.symbols.lookup("ids").type
    assert values.dtype == "fp32" and values.shape == ["4"]
    assert ids.dtype == "int32" and ids.shape == ["4"]


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_the_values_keep_the_input_dtype(dtype):
    pipeline = compile_program(program(dtype, [4, 8], "argmin_last"))
    assert pipeline.symbols.lookup("values").type.dtype == dtype
    assert pipeline.symbols.lookup("ids").type.dtype == "int32"


@pytest.mark.parametrize("op", OPERATORS_UNDER_TEST)
def test_an_unused_values_result_costs_no_buffer_but_still_leaves_a_producer(op):
    source = f"""
pipeline only_ids {{
 input x:fp32[4,8]
 values, ids = {op}(x)
 output ids
}}
"""
    pipeline = compile_program(source)
    assert pipeline.triton.complete
    (kernel,) = reduce_kernels(pipeline)
    assert "values_ptr" not in kernel.source
    assert "ids_ptr" in kernel.source
    produced = run_reference(pipeline.reference)
    assert all(0 <= column < 8 for column in produced["ids"])


@pytest.mark.parametrize("op", OPERATORS_UNDER_TEST)
def test_an_unused_indices_result_costs_no_buffer_but_still_leaves_a_producer(op):
    """The other half of the optional-result policy: a caller may want only the
    winning values, and the kernel must then not mention the index buffer."""
    source = f"""
pipeline only_values {{
 input x:fp32[4,8]
 values, ids = {op}(x)
 output values
}}
"""
    pipeline = compile_program(source)
    assert pipeline.triton.complete
    (kernel,) = reduce_kernels(pipeline)
    assert "ids_ptr" not in kernel.source
    assert "values_ptr" in kernel.source
    produced = run_reference(pipeline.reference)
    assert len(produced["values"]) == 4


def test_there_is_no_axis_argument_surface():
    """The operator is last-axis *by name*; a `dim`/`axis` argument would be a
    different feature (`argmax_axis`), so the call site is refused rather than
    silently ignoring the axis."""
    source = program("fp32", [4, 8], "argmax_last").replace(
        "argmax_last(x)", "argmax_last(x, 0)")
    with pytest.raises(CompileError, match="Operator 'argmax_last' expects 1 arguments") as caught:
        compile_program(source)
    assert "expects 1 arguments" in str(caught.value)


# --------------------------------------------------------------------------
# static rejections
# --------------------------------------------------------------------------
def test_a_third_name_is_refused():
    source = program("fp32", [4, 8], "argmax_last").replace(
        "values, ids = argmax_last(x)", "values, ids, extra = argmax_last(x)")
    with pytest.raises(CompileError, match='argmax_last returns values and indices') as caught:
        compile_program(source)
    assert "needs two names" in str(caught.value)


def test_a_rank_one_input_is_refused_because_a_scalar_is_not_a_tensor():
    with pytest.raises(CompileError, match='reducing the last axis of a rank-1 input gives a scalar') as caught:
        compile_program(program("fp32", [8], "argmin_last"))
    assert "at least two dimensions" in str(caught.value)


def test_a_symbolic_last_axis_is_refused():
    source = program("fp32", [4, 8], "argmax_last").replace(
        " input x:fp32[4,8]", " input x:fp32[4,D]")
    with pytest.raises(CompileError, match='the row tile is a constexpr and the grid is computed from the extents') as caught:
        compile_program(source)
    assert "literal last axis" in str(caught.value)


def test_a_symbolic_leading_extent_is_refused():
    source = program("fp32", [4, 8], "argmax_last").replace(
        " input x:fp32[4,8]", " input x:fp32[N,8]")
    with pytest.raises(CompileError, match='argmax_last requires literal leading extents so the grid is static') as caught:
        compile_program(source)
    assert "literal leading extents" in str(caught.value)


@pytest.mark.parametrize("rule", [infer_argmax_last, infer_argmin_last])
def test_an_empty_last_axis_is_a_static_error(rule):
    """Zero cannot be *declared* (`tensor dimensions must be positive`), so the
    rule is checked at the rule boundary where a future producer of a zero
    extent would meet it."""
    with pytest.raises(RuleError, match='_last requires a non-empty last axis') as caught:
        rule([TensorType("fp32", ["4", "0"])], 2)
    assert "non-empty last axis" in str(caught.value)


def test_a_zero_extent_cannot_even_be_declared():
    """The other half of the empty-axis contract: the declaration pass refuses
    the shape before any rule sees it."""
    with pytest.raises(CompileError, match='tensor dimensions must be positive') as caught:
        compile_program(program("fp32", [4, 0], "argmax_last"))
    assert "must be positive" in str(caught.value)


# --------------------------------------------------------------------------
# numerics against an independent oracle
# --------------------------------------------------------------------------
@pytest.mark.parametrize("op", OPERATORS_UNDER_TEST)
@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("shape", [[4, 8], [3, 5], [2, 3, 7], [2, 1]])
def test_values_and_indices_match_an_independent_oracle(op, dtype, shape):
    pipeline = compile_program(program(dtype, shape, op))
    data = feeds(pipeline)["x"]
    produced = run_reference(pipeline.reference)
    expected_values, expected_indices = oracle(
        data, shape[-1], maximum=(op == "argmax_last"))
    assert list(produced["ids"]) == expected_indices
    assert list(produced["values"]) == pytest.approx(
        expected_values, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("op", OPERATORS_UNDER_TEST)
def test_a_masked_tail_and_a_static_chunk_loop_still_match_the_oracle(op):
    """Width 13 is not a power of two and the platform budget is 4, so the
    launch is four chunks with a one-lane last chunk: the masked tail and the
    cross-chunk merge are both exercised."""
    config = replace(PLATFORMS["cuda"], reduction_block_limit=4)
    pipeline = compile_program(program("fp32", [2, 13], op), platform_config=config)
    (kernel,) = reduce_kernels(pipeline)
    constants = {name.split(":")[0]: argument.text
                 for name, argument in zip(kernel.parameters, kernel.launch)}
    assert constants["CHUNK"] == "4" and constants["CHUNKS"] == "4"
    assert "tl.static_range(0, CHUNKS)" in kernel.source
    data = feeds(pipeline)["x"]
    produced = run_reference(pipeline.reference)
    expected_values, expected_indices = oracle(
        data, 13, maximum=(op == "argmax_last"))
    assert list(produced["ids"]) == expected_indices
    assert list(produced["values"]) == pytest.approx(
        expected_values, rel=1e-6, abs=1e-6)


# --------------------------------------------------------------------------
# the four semantic decisions, on hand-built rows
# --------------------------------------------------------------------------
@pytest.mark.parametrize("op", OPERATORS_UNDER_TEST)
def test_an_all_equal_row_returns_index_zero(op):
    pipeline = compile_program(program("fp32", [3, 4], op))
    produced = execute(pipeline, [[7.0, 7.0, 7.0, 7.0,
                                   3.0, 3.0, 3.0, 3.0,
                                   0.0, 0.0, 0.0, 0.0]])
    assert list(produced["ids"]) == [0, 0, 0]


def test_equal_extremes_take_the_smaller_index_not_the_first_encountered():
    """The tie rule is a merge rule, not an accident of scan order: for argmax
    the equal maxima are at columns 1 and 3; for argmin the equal minima are at
    columns 2 and 4. Both must keep the smaller column."""
    top = compile_program(program("fp32", [2, 5], "argmax_last"))
    produced = execute(top, [[1.0, 9.0, 2.0, 9.0, 3.0,
                              4.0, 4.0, 4.0, 4.0, 0.0]])
    assert list(produced["ids"]) == [1, 0]

    bottom = compile_program(program("fp32", [2, 5], "argmin_last"))
    produced = execute(bottom, [[1.0, -9.0, -9.0, 0.0, 2.0,
                                 1.0, 0.0, 1.0, 0.0, 2.0]])
    got = list(produced["ids"])
    # row 0: -9 at columns 1 and 2 -> 1; row 1: 0 at columns 1 and 3 -> 1
    assert got == [1, 1]


@pytest.mark.parametrize("op, rows, expected_indices, expected_values", [
    ("argmax_last",
     [[1.0, float("nan"), 3.0, float("nan")],
      [float("nan"), float("nan"), float("nan"), float("nan")]],
     [2, 0], [3.0, -math.inf]),
    ("argmin_last",
     [[float("nan"), 2.0, float("nan"), 3.0],
      [float("nan"), float("nan"), float("nan"), float("nan")]],
     [1, 0], [2.0, math.inf]),
])
def test_nan_is_never_selected_and_an_all_nan_row_returns_the_identity(
        op, rows, expected_indices, expected_values):
    """NaN is not a candidate. A row with no candidate returns the direction's
    identity with index 0 -- stated, because torch's `argmax` would select NaN
    instead."""
    pipeline = compile_program(program("fp32", [len(rows), 4], op))
    produced = execute(pipeline, [[value for row in rows for value in row]])
    assert list(produced["ids"]) == expected_indices
    got = list(produced["values"])
    assert all(
        (math.isinf(value) and math.isinf(want) and value == want)
        or value == pytest.approx(want)
        for value, want in zip(got, expected_values))


@pytest.mark.parametrize("op, rows, expected_indices, expected_values", [
    ("argmax_last",
     [[1.0, math.inf, 2.0, math.inf],
      [-math.inf, -math.inf, -math.inf, -math.inf]],
     [1, 0], [math.inf, -math.inf]),
    ("argmin_last",
     [[1.0, -math.inf, 2.0, -math.inf],
      [math.inf, math.inf, math.inf, math.inf]],
     [1, 0], [-math.inf, math.inf]),
])
def test_plus_minus_inf_follow_the_normal_max_min_rules(
        op, rows, expected_indices, expected_values):
    """Unlike `topk_last`, where -inf is padding and never selected, here an
    infinite extreme is a real candidate; the smallest index still wins."""
    pipeline = compile_program(program("fp32", [len(rows), 4], op))
    produced = execute(pipeline, [[value for row in rows for value in row]])
    assert list(produced["ids"]) == expected_indices
    assert [value for value in produced["values"]] == expected_values


# --------------------------------------------------------------------------
# the lowering: one launch, static chunk loop, affine stores, portable shape
# --------------------------------------------------------------------------
def test_the_lowering_is_one_launch_with_affine_stores_outside_the_loop():
    """The three measured portability constraints, pinned as text:
    one launch; stores are `out + row` (scalar base, no lane term); the carried
    pair is written once, after the static chunk loop."""
    pipeline = compile_program(program("fp32", [4, 8], "argmax_last"))
    (kernel,) = reduce_kernels(pipeline)
    assert kernel.grid == "(4,)"
    assert pipeline.triton.complete

    addresses = store_addresses(kernel.source)
    assert len(addresses) == 2
    assert {address.target for address in addresses} == {"values_ptr", "ids_ptr"}
    for address in addresses:
        assert address.verdict == "affine"
        assert address.expression == f"{address.target} + row"
        assert "*" not in address.expression and "//" not in address.expression
    assert address_violations(kernel.source) == []

    tree = ast.parse(kernel.source)
    function = tree.body[0]
    loops = [statement for statement in function.body
             if isinstance(statement, ast.For)]
    assert len(loops) == 1
    assert "static_range" in kernel.source
    # Every store is a top-level statement after the loop -- none inside it.
    stores = [statement for statement in function.body
              if isinstance(statement, ast.Expr)
              and isinstance(statement.value, ast.Call)
              and getattr(statement.value.func, "attr", "") == "store"]
    assert len(stores) == 2
    assert all(function.body.index(store) > function.body.index(loops[0])
               for store in stores)
    assert not [node for node in ast.walk(loops[0])
                if isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "store"]


def test_the_lowering_emits_no_two_dimensional_tile():
    pipeline = compile_program(program("fp32", [4, 8], "argmax_last"))
    (kernel,) = reduce_kernels(pipeline)
    assert "[:, None]" not in kernel.source
    assert "None]" not in kernel.source
    assert kernel.source.count("tl.arange(") == 1


@pytest.mark.parametrize("op", OPERATORS_UNDER_TEST)
def test_the_emitted_source_is_inside_the_measured_kunlun_contract(op):
    """The contract fields are reused, not branched on: the same source must
    pass `store_inside_loop_ok=False`, `gather_load_ok=False` and
    `affine_write_addresses=True`."""
    pipeline = compile_program(program("fp32", [4, 8], op))
    (kernel,) = reduce_kernels(pipeline)
    assert contract_violations(kernel.source, PLATFORMS["kunlunxin"]) == []
    assert address_violations(kernel.source) == []


def test_the_chunk_width_comes_from_the_platform_reduction_budget():
    """A 昆仑-profile target with width 300 and its measured 128-lane budget
    must emit three chunks of 128; the emitter reads the record, never the
    platform name."""
    config = replace(PLATFORMS["kunlunxin"],
                     allowed_kernel_kinds=("pointwise", "reduce_with_index"))
    pipeline = compile_program(program("fp32", [2, 300], "argmax_last"),
                               platform_config=config)
    assert pipeline.triton.complete
    (kernel,) = reduce_kernels(pipeline)
    constants = {name.split(":")[0]: argument.text
                 for name, argument in zip(kernel.parameters, kernel.launch)}
    assert constants["CHUNK"] == "128"
    assert constants["CHUNKS"] == "3"
    assert contract_violations(kernel.source, config) == []


def test_a_large_leading_row_count_is_strided_inside_the_kernel():
    """A grid beyond the target's ceiling is not launched.

    The generated CANN ArgMax variants reach ~10^6 leading rows; a target that
    refuses a grid above 65535 has to loop rows inside the kernel (the same
    contract the pointwise emitter keeps).  Small row counts keep the
    one-row-per-program source byte-for-byte, which is what the other tests pin.
    """
    from flowgpu.compiler.backend.platform_configs import resolve_platform

    kinds = ('pointwise', 'rowwise', 'primitive', 'layout', 'transpose', 'rope',
             'reduce_with_index')
    config = resolve_platform('ascend-910b', device_cores=32,
                              allowed_kernel_kinds=kinds)

    big = '''
pipeline big {
    input x : fp32[2,7,256,251]
    xf = to_fp32(x)
    r0 = move_first_last(xf)
    values, indices = argmax_last(r0)
    b0 = move_first_last(indices)
    b1 = move_first_last(b0)
    y = cast_to(b1, int64)
    output y
}
'''
    kernels = compile_source(big, platform_config=config).pipelines[0].triton.kernels
    reducer = next(k for k in kernels if k.kind == 'reduce_with_index')
    assert reducer.grid == '(65535,)'
    assert 'tl.num_programs(0)' in reducer.source
    assert 'ROWS: tl.constexpr' in reducer.source

    small = '''
pipeline small {
    input x : fp32[8,16]
    xf = to_fp32(x)
    values, indices = argmax_last(xf)
    y = cast_to(indices, int64)
    output y
}
'''
    kernels = compile_source(small, platform_config=config).pipelines[0].triton.kernels
    reducer = next(k for k in kernels if k.kind == 'reduce_with_index')
    assert reducer.grid == '(8,)'
    assert 'tl.num_programs(0)' not in reducer.source
