"""`topk_last` and `gather_last`: the router/gate family, both halves.

The content census put this family first among the gaps that remained after the
value layer (10 tasks). The tests below are grouped by the claim each one holds:
the type rules, the emission, the reference semantics against an independent
oracle, and the three semantic decisions the operator makes rather than inherits
(tie order, NaN, and what a short row pads with).
"""
import math

import pytest

from compiler.analysis.access_shape import BLOCK, GATHER, of_op
from compiler.backend.platform_configs import PLATFORMS
from compiler.backend.reference import inputs_for, run as run_reference
from compiler.compile import compile_source
from compiler.frontend.operator import OPERATORS
from compiler.frontend.diagnostic import CompileError

ROUTER = """
pipeline router {
 input x:fp32[4,8]
 capped = tanh(x)
 probs = softmax(capped)
 values, ids = topk_last(capped, 2)
 weights = gather_last(probs, ids)
 output weights
 output ids
}
"""


def compile_program(source=ROUTER):
    return compile_source(source).pipelines[0]


def feeds(pipeline):
    return {name: values for name, _, _, values in inputs_for(pipeline.reference, {})}


# --------------------------------------------------------------------------
# the registry and the type rules
# --------------------------------------------------------------------------
def test_topk_is_registered_as_a_multi_result_operator():
    assert OPERATORS["topk_last"].multi is True
    assert OPERATORS["topk_last"].pointwise is False
    assert OPERATORS["topk_last"].acc is None


def test_the_two_results_are_a_values_tensor_and_an_int32_index_tensor():
    """Read from the symbol table: a program need not *output* both results, and
    a router usually outputs neither of `values` -- it only needs the indices."""
    pipeline = compile_program()
    values = pipeline.symbols.lookup("values").type
    indices = pipeline.symbols.lookup("ids").type
    assert values.dtype == "fp32" and values.shape == ["4", "2"]
    assert indices.dtype == "int32" and indices.shape == ["4", "2"]


def test_an_unused_result_costs_no_buffer_but_still_leaves_a_producer():
    """`values` is dead in the router shape, and 13.7 gives a dead value no
    buffer. The kernel must still run for `ids`, so the emitter stores only what
    has a slot instead of dropping the kernel and leaving `ids` unproduced."""
    pipeline = compile_program()
    assert pipeline.triton.complete
    topk = next(kernel for kernel in pipeline.triton.kernels if kernel.kind == "topk")
    assert "values_ptr" not in topk.source
    assert "ids_ptr" in topk.source
    produced = run_reference(pipeline.reference)
    assert all(0 <= column < 8 for column in produced["ids"])


def test_a_third_name_for_topk_is_refused():
    source = ROUTER.replace("values, ids = topk_last(capped, 2)",
                            "values, ids, extra = topk_last(capped, 2)")
    with pytest.raises(CompileError, match='topk_last returns values and indices') as caught:
        compile_program(source)
    assert "two names" in str(caught.value)


def test_a_symbolic_k_is_refused_because_the_output_extent_must_be_writable():
    source = ROUTER.replace("topk_last(capped, 2)", "topk_last(capped, k)")
    source = source.replace(" input x:fp32[4,8]", " input x:fp32[4,8] input k:int32[1]")
    with pytest.raises(CompileError, match='topk_last requires a static integer k'):
        compile_program(source)


def test_a_k_beyond_the_last_axis_is_refused():
    source = ROUTER.replace("topk_last(capped, 2)", "topk_last(capped, 99)")
    with pytest.raises(CompileError, match='99 exceeds the last axis') as caught:
        compile_program(source)
    assert "exceeds the last axis" in str(caught.value)


def test_a_symbolic_last_axis_is_refused():
    source = ROUTER.replace(" input x:fp32[4,8]", " input x:fp32[4,D]")
    with pytest.raises(CompileError, match='the row tile is a constexpr and the grid is computed from the extents') as caught:
        compile_program(source)
    assert "literal last axis" in str(caught.value)


def test_gather_last_needs_an_integer_index_of_the_same_rank_and_leading_dims():
    bad_rank = ROUTER.replace("gather_last(probs, ids)", "gather_last(probs, x)")
    with pytest.raises(CompileError, match='gather_last requires an integer index') as caught:
        compile_program(bad_rank)
    assert "integer index" in str(caught.value)

    float_index = ROUTER.replace("gather_last(probs, ids)", "gather_last(probs, capped)")
    with pytest.raises(CompileError, match='gather_last requires an integer index') as caught:
        compile_program(float_index)
    assert "integer index" in str(caught.value)


# --------------------------------------------------------------------------
# emission: what the two kernels actually do
# --------------------------------------------------------------------------
def test_the_topk_kernel_selects_the_lower_column_on_a_tie():
    """`torch.argsort` leaves tie order unspecified, so the language states its
    own: equal values go to the smaller column. The emitted form is what makes
    that true -- the index is the *minimum* column reaching the maximum, not
    whatever lane a reduction tree happens to surface."""
    source = compile_program().triton.kernels
    kernel = next(kernel for kernel in source if kernel.kind == "topk")
    assert "tl.min(tl.where(" in kernel.source
    assert "WIDTH" in kernel.source and "K" in kernel.source
    assert kernel.grid == "(4,)"


def test_the_topk_kernel_masks_the_selected_lane_out_for_the_next_round():
    kernel = next(kernel for kernel in compile_program().triton.kernels
                  if kernel.kind == "topk")
    assert 'float("-inf")' in kernel.source
    assert "for step in range(K)" in kernel.source


def test_the_gather_kernel_stores_under_exactly_the_load_mask():
    """The measured 昆仑 contract: a store mask broader than the load mask is what
    breaks there. The zero region is owned by its own store, and the gathered
    values are written under the same predicate they were loaded with."""
    kernel = next(kernel for kernel in compile_program().triton.kernels
                  if kernel.kind == "axis_gather")
    lines = [line.strip() for line in kernel.source.splitlines() if "tl.store" in line]
    assert any("mask=inside" in line for line in lines)
    assert any("mask=in_bounds" in line for line in lines)
    assert "source_position" not in kernel.source or "row * WIDTH + slot" in kernel.source


def test_gather_last_is_a_gather_access_and_topk_is_a_block_access():
    """`gather_last` selects by a value inside the row, so it is the same class of
    access as a row gather and the platform gate treats it that way; `topk_last`
    reads a whole row and is axis-local."""
    assert of_op("gather_last").kind == GATHER
    assert of_op("topk_last").kind == BLOCK


def test_the_router_plan_is_refused_before_emission_on_kunlunxin():
    from compiler.analysis.access_shape import group_conflicts

    pipeline = compile_program()
    conflicts = group_conflicts(pipeline.fusion, PLATFORMS["kunlunxin"])
    assert conflicts and "gather" in conflicts[0]


# --------------------------------------------------------------------------
# reference semantics against an independent oracle
# --------------------------------------------------------------------------
def _oracle(x, width, k):
    """Descending top-k with the language's tie rule, computed from scratch."""
    rows = len(x) // width
    values, indices = [], []
    for row in range(rows):
        candidates = []
        for column in range(width):
            value = float(x[row * width + column])
            if value == value and value != float("-inf"):
                candidates.append((value, column))
        candidates.sort(key=lambda entry: (-entry[0], entry[1]))
        for slot in range(k):
            chosen = candidates[slot] if len(candidates) > slot else (0.0, 0)
            values.append(chosen[0])
            indices.append(chosen[1])
    return values, indices


def test_topk_matches_an_independent_oracle():
    """Both results, from a program that outputs both: values and indices are
    produced by the same argmax loop, so a test that checked only the indices
    would miss a swapped pair."""
    source = """
pipeline pick {
 input x:fp32[4,8]
 values, ids = topk_last(x, 3)
 output values
 output ids
}
"""
    pipeline = compile_program(source)
    produced = run_reference(pipeline.reference)
    expected_values, expected_indices = _oracle(feeds(pipeline)["x"], 8, 3)
    assert list(produced["ids"]) == expected_indices
    for got, want in zip(produced["values"], expected_values):
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6)


def test_the_router_oracle_agrees_on_the_selected_columns():
    """The observable half of the router: which experts each token picks, from an
    independent reimplementation of tanh -> softmax -> top-k."""
    pipeline = compile_program()
    produced = run_reference(pipeline.reference)
    capped = [math.tanh(float(v)) for v in feeds(pipeline)["x"]]
    _, expected_indices = _oracle(capped, 8, 2)
    assert list(produced["ids"]) == expected_indices


@pytest.mark.parametrize("row, expected", [
    ([5.0, 5.0, 1.0], [0, 1]),          # tie -> the lower column first
    ([1.0, 5.0, 5.0], [1, 2]),
    ([-float("inf"), 3.0, float("nan")], [1, 0]),   # NaN and -inf are not candidates
    ([float("nan"), float("nan"), float("nan")], [0, 0]),   # no candidate: pad 0/0
])
def test_the_tie_nan_and_padding_rules(row, expected):
    """A hand-built row, because the generated input data has no ties or NaNs and
    the three rules are exactly what a device and a reference must agree on."""
    values, indices = _oracle(row, 3, 2)
    assert indices[:2] == expected


def test_gather_last_normalises_a_negative_index_and_reads_zero_out_of_bounds():
    source = """
pipeline g {
 input x:fp32[2,4] input index:int32[2,3]
 y = gather_last(x, index)
 output y
}
"""
    pipeline = compile_program(source)
    data = feeds(pipeline)
    produced = run_reference(pipeline.reference)["y"]
    x, index = data["x"], data["index"]
    expected = []
    for flat in range(6):
        row, _ = divmod(flat, 3)
        slot = index[flat]
        if slot < 0:
            slot += 4
        expected.append(x[row * 4 + slot] if 0 <= slot < 4 else 0.0)
    assert produced == pytest.approx(expected)


# --------------------------------------------------------------------------
# the router example is a compilable program, not a sketch
# --------------------------------------------------------------------------
def test_the_router_example_compiles_and_emits_both_kernels():
    source = open("examples/batch10/moe_router_topk.fgpu", encoding="utf-8").read()
    pipeline = compile_source(source).pipelines[0]
    kinds = [kernel.kind for kernel in pipeline.triton.kernels]
    assert "topk" in kinds and "axis_gather" in kinds
    assert pipeline.triton.complete

    produced = run_reference(pipeline.reference)
    assert len(produced["ids"]) == 8        # [4 rows, k=2]
    assert len(produced["weights"]) == 8
    for ids in (produced["ids"][i:i + 2] for i in range(0, 8, 2)):
        assert ids[0] != ids[1]             # top-k of distinct columns


def test_the_single_step_ssm_update_is_expressible_with_the_existing_operators():
    """Task36 was filed under `recurrence`, and it is not one: the reference takes
    the current state and does one step. Every operator it needs exists, so it is
    a program rather than an operator request -- which is why the census moved it.

    The oracle recomputes `dA = exp(dt*A)`, `S' = dA*S + dt*B*x`, `y = S'·C + D*x`
    from scratch, in double precision, on the same generated inputs.
    """
    source = open("examples/batch10/selective_state_update.fgpu", encoding="utf-8").read()
    pipeline = compile_source(source).pipelines[0]
    assert pipeline.triton.complete
    produced = run_reference(pipeline.reference)
    data = feeds(pipeline)

    batch, heads, dim, dstate = 2, 2, 4, 8

    def at(values, shape, *indices):
        flat = 0
        for index, extent in zip(indices, shape):
            flat = flat * extent + index
        return values[flat]

    expected_state, expected_y = [], []
    for b in range(batch):
        for h in range(heads):
            for p in range(dim):
                step = math.log1p(math.exp(at(data["dt"], (batch, heads, dim), b, h, p)))
                total = 0.0
                for n in range(dstate):
                    decay = math.exp(step * at(data["A"], (heads, dstate), h, n))
                    state = at(data["state"], (batch, heads, dim, dstate), b, h, p, n)
                    driven = (step * at(data["B"], (batch, heads, dstate), b, h, n)
                              * at(data["x"], (batch, heads, dim), b, h, p))
                    updated = decay * state + driven
                    expected_state.append(updated)
                    total += updated * at(data["C"], (batch, heads, dstate), b, h, n)
                expected_y.append(total + at(data["D"], (batch, heads, dim), b, h, p)
                                  * at(data["x"], (batch, heads, dim), b, h, p))

    assert max(abs(got - want) for got, want in zip(produced["updated"], expected_state)) < 1e-5
    assert max(abs(got - want) for got, want in zip(produced["y"], expected_y)) < 1e-5
