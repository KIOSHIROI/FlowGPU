"""21.6's precision contract.

The dimension is measured by the completeness of a declaration: the number of
reductions that did not say what they accumulate in. That number is zero by
construction once the declaration is a required argument, so what these tests
hold is the construction itself -- the argument cannot be omitted, the floor is
enforced at the call site, and `acc` stays out of the types.

The last one is the claim that keeps the two precisions honest. If `acc` leaked
into type inference then declaring an accumulator would silently change every
inferred type, and the split in 21.6 would collapse back into one number.
"""

import json

import pytest

from flowgpu.compiler.backend.value_layer import NAMES as VALUE_LAYER_NAMES
from flowgpu.compiler.cli import main
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.diagnostic import CompileError
from flowgpu.compiler.operator import OPERATORS, infer_unary_same_type, register_operator
from flowgpu.compiler.precision import ACC_FLOOR, declared, declined, dtype_class, width

NARROW_ACCUMULATOR = """
pipeline narrow {
    input x : bf16[N,D]
    y = mean(x)
    output y
}
"""

BOOL_PIPELINES = [
    "input x : bool[N,D]\n    y = add(x, x)",
    "input x : bool[N,D]\n    y = silu(x)",
    "input x : bool[N,D]\n    input w : bool[N,N]\n    y = matmul(x, w)",
]

INT32 = """
pipeline integers {
    input x : int32[N,D]
    input y : int32[N,D]
    z = add(x, y)
    output z
}
"""


@pytest.fixture
def registry():
    """Restore the operator table, whatever a test registers into it."""
    saved = dict(OPERATORS)

    yield

    OPERATORS.clear()
    OPERATORS.update(saved)


def test_every_operator_states_a_position():
    """Rule 1's consequence: nothing is silently in neither group."""
    anchors = [name for name, spec in OPERATORS.items() if spec.pointwise is False]

    assert set(declared(OPERATORS)) | set(declined(OPERATORS)) == set(anchors)
    assert not set(declared(OPERATORS)) & set(declined(OPERATORS))

    # A pointwise operator that accumulated would contradict itself: relu sums
    # over nothing.
    for name, spec in OPERATORS.items():
        if spec.pointwise is True:
            assert spec.acc is None, name


def test_the_reductions_that_accumulate_are_named():
    """21.6 names exactly these three: matmul's K, softmax's sum, mean(x^2)."""
    assert declared(OPERATORS) == ("rms_norm", "rmsnorm_hf", "matmul", "softmax", "attention", "causal_attention", "cumsum", "cumsum_totals", "cummin_last", "cummax_last", "segmented_cumsum", "segmented_matmul", "segmented_lora_matmul", "linear_decode_forward", "sum_last", "max_last", "mean_last", "layer_norm", "batch_norm", "conv2d", "conv2d_same", "conv2d_stride2", "conv2d_stride2_same", "conv2d_ex", "sum_axis", "max_axis", "cumsum_axis", "softmax_axis", "rms_norm_axis", "gated_delta_recurrence")


def test_the_anchor_that_accumulates_nothing_is_named():
    """`transpose` is an anchor and a permutation: it sums over nothing.

    This is the limit of the metric worth stating: "is a reduction" cannot be
    inferred from the registry, so the contract makes every operator state a
    position and keeps both groups short enough to read. The v0.2 layout
    operators (reshape/split/concat) accumulate nothing either -- they move
    elements, they do not combine them.
    """
    assert declined(OPERATORS) == ("transpose", "rope", "reshape", "split", "concat", "pad_cat", "patchify", "scatter_by_index") + ("exp", "exp2", "log", "log2", "neg_inf_like", "ones_like", "zeros_like", "to_fp32", "isfinite", "where", "broadcast_like", "cast_like", "cast_to", "move_first_last", "gather_row_offset", "gather_rows", "store_cache_4d", "div", "sort_last") + VALUE_LAYER_NAMES + ("topk_last", "gather_last", "argmax_last", "argmin_last", "segment_rank", "interleave")


@pytest.mark.parametrize("name", ["rms_norm", "matmul", "softmax"])
def test_no_declared_accumulator_is_below_the_floor(name):
    assert width(OPERATORS[name].acc) >= width(ACC_FLOOR)


def test_a_required_declaration_cannot_be_omitted(registry):
    """Rule 1, enforced where it cannot be forgotten: the signature."""
    with pytest.raises(TypeError, match="register_operator\\(\\) missing 1 required keyword-only argument: 'acc'") as refusal:
        register_operator("forgotten", 1, infer_unary_same_type, pointwise=False)

    assert "acc" in str(refusal.value)
    assert "forgotten" not in OPERATORS


def test_a_float_reduction_may_not_accumulate_in_bf16(registry):
    """Rule 2, at the call site where the operand dtype is known."""
    register_operator(
        "mean", 1, infer_unary_same_type, pointwise=False, acc="bf16"
    )

    with pytest.raises(CompileError, match="olumn 9: Operator 'mean' accumulates float inputs in bf16, which is narrower than fp32: 21\\.6 requires the accumulator of a reduction over floats to be at least fp32") as refusal:
        compile_source(NARROW_ACCUMULATOR)

    assert refusal.value.kind.value == "type"
    assert "narrower than fp32" in refusal.value.message
    assert "21.6" in refusal.value.message


def test_acc_does_not_propagate_but_store_does(registry):
    """Rule 3.

    `h` is stored bf16 and accumulated fp32 at the same time -- which is the
    whole reason the two are separate numbers. The relu after it inherits the
    *store* and declares no accumulator of its own, because it has nothing to
    accumulate.
    """
    result = compile_source(open("examples/mlp.fgpu").read()).pipelines[0]
    graph = result.dataflow

    # `h` is the matmul's own result: stored bf16, accumulated fp32. That pair
    # is the reason these are two numbers rather than one.
    assert OPERATORS["matmul"].acc == "fp32"
    assert graph.values["h"].dtype == "bf16"

    # `h2` is silu(h). silu accumulates nothing, so it declares nothing, and the
    # accumulator of the matmul does not travel through it -- but the *store*
    # does: h2 is bf16 because h was.
    silu = next(node for node in graph.nodes if node.op == "silu")

    assert OPERATORS[silu.op].acc is None
    assert graph.values[silu.output].dtype == graph.values[silu.inputs[0]].dtype
    assert graph.values[silu.output].dtype == "bf16"


@pytest.mark.parametrize("body", BOOL_PIPELINES)
def test_bool_is_not_a_number(body):
    """21.6's dtype classes, which are the degenerate case of this dimension.

    README's known gap 2 was that a rule comparing only operands accepts
    `bool + bool`: the operands match and the computation is meaningless.
    """
    source = f"pipeline b {{\n    {body}\n    output y\n}}\n"

    with pytest.raises(CompileError, match='a bool: bool is not a number') as refusal:
        compile_source(source)

    assert refusal.value.kind.value == "type"
    assert "bool is not a number" in refusal.value.message


def test_a_permutation_of_bool_is_still_fine():
    """The class rule is about arithmetic, not about bool being unusable."""
    source = """
pipeline b {
    input x : bool[N,D]
    y = transpose(x)
    output y
}
"""
    result = compile_source(source).pipelines[0]

    assert result.dataflow.values["y"].shape == ["D", "N"]


def test_integers_are_numbers():
    result = compile_source(INT32).pipelines[0]

    assert result.dataflow.values["z"].dtype == "int32"
    assert dtype_class("int32") == "integer"


def test_bool_and_float_are_different_classes():
    assert dtype_class("bf16") == "float"
    assert dtype_class("bool") == "bool"


def test_the_view_shows_both_precisions_per_value(capsys):
    main(["examples/residual_norm.fgpu", "--emit", "precision"])
    printed = capsys.readouterr().out

    assert "y            store bf16   acc -" in printed
    assert "z            store bf16   acc fp32" in printed
    assert "accumulate   rms_norm     fp32" in printed
    assert "declined     transpose    accumulates nothing" in printed
    assert "every float accumulation is at least 32 bits" in printed


def test_the_json_carries_the_contract(capsys):
    main(["examples/mlp.fgpu", "--emit", "json"])
    precision = json.loads(capsys.readouterr().out)["pipelines"][0]["precision"]

    assert precision["floor"] == "fp32"
    assert precision["declared"] == {
        "rms_norm": "fp32",
        "rmsnorm_hf": "fp32",
        "matmul": "fp32",
        "softmax": "fp32",
        "attention": "fp32",
        "causal_attention": "fp32",
        "cumsum": "fp32",
        "cumsum_totals": "fp32",
        "cummin_last": "fp32",
        "cummax_last": "fp32",
        "segmented_cumsum": "fp32",
        "segmented_matmul": "fp32",
        "segmented_lora_matmul": "fp32",
        "linear_decode_forward": "fp32",
        "sum_last": "fp32",
        "max_last": "fp32",
        "mean_last": "fp32",
        "layer_norm": "fp32",
        "batch_norm": "fp32",
        "conv2d": "fp32",
        "conv2d_same": "fp32",
        "conv2d_stride2": "fp32",
        "conv2d_stride2_same": "fp32",
        "conv2d_ex": "fp32",
        "sum_axis": "fp32",
        "max_axis": "fp32",
        "cumsum_axis": "fp32",
        "softmax_axis": "fp32",
        "rms_norm_axis": "fp32",
        "gated_delta_recurrence": "fp32",
    }
    assert precision["declined"] == list(("transpose", "rope", "reshape", "split", "concat", "pad_cat", "patchify", "scatter_by_index") + ("exp", "exp2", "log", "log2", "neg_inf_like", "ones_like", "zeros_like", "to_fp32", "isfinite", "where", "broadcast_like", "cast_like", "cast_to", "move_first_last", "gather_row_offset", "gather_rows", "store_cache_4d", "div", "sort_last") + VALUE_LAYER_NAMES + ("topk_last", "gather_last", "argmax_last", "argmin_last", "segment_rank", "interleave"))

    stores = {entry["value"]: entry["store"] for entry in precision["values"]}
    accs = {entry["value"]: entry["acc"] for entry in precision["values"]}

    assert stores["h"] == "bf16"
    assert accs["h"] == "fp32"
    assert accs["h2"] is None   # silu sums over nothing
    assert accs["x"] is None    # an input has no producer to accumulate


SOFTMAX_PROGRAM = """
pipeline sm {
    input x : bf16[N,D]
    y = softmax(x)
    output y
}
"""

RMS_NORM_PROGRAM = """
pipeline rn {
    input x : bf16[N,D]
    input weight : bf16[D]
    y = rms_norm(x, weight)
    output y
}
"""


def test_the_backend_widens_the_reduction_it_declared(tmp_path, capsys):
    """The widening is in the emitted module, and it says where it came from.

    That the emitter *reads* the declaration is proved by the refusal test
    below: a hardcoded tl.float32 would have lowered a reduction that declared
    no accumulator at all, instead of refusing. This test holds the other half,
    that the widening is still there and still attributed to acc.
    """
    path = tmp_path / "sm.fgpu"
    path.write_text(SOFTMAX_PROGRAM)

    assert main([str(path), "--emit", "triton"]) == 0

    printed = capsys.readouterr().out
    assert ".to(tl.float32)" in printed
    assert "declared acc" in printed


def test_a_declaration_the_compiler_cannot_measure_says_so(registry):
    """A width it does not know used to arrive as a KeyError."""
    import dataclasses

    OPERATORS["softmax"] = dataclasses.replace(OPERATORS["softmax"], acc="fp64")

    with pytest.raises(AssertionError, match="Operator 'softmax' declares acc='fp64', which is not a dtype this compiler knows; 21\\.6 compares precisions by width, so a declaration it cannot measure is a declaration it cannot honour") as refusal:
        compile_source(SOFTMAX_PROGRAM)

    assert "not a dtype this compiler knows" in str(refusal.value)


def test_a_reduction_without_a_declaration_cannot_be_lowered(registry, tmp_path):
    """The other direction: no declaration, no code.

    21.6 says an undeclared accumulation is worse than a wrong one, so the
    backend refuses to invent a dtype. This is a compiler bug rather than a
    program's problem, which is why it raises instead of becoming a diagnostic.
    """
    import dataclasses

    OPERATORS["rms_norm"] = dataclasses.replace(OPERATORS["rms_norm"], acc=None)

    path = tmp_path / "rn.fgpu"
    path.write_text(RMS_NORM_PROGRAM)

    with pytest.raises(AssertionError, match='rms_norm is lowered as a reduction but declares no accumulator: 21\\.6 requires one, and this backend will not invent it') as refusal:
        main([str(path), "--emit", "triton"])

    assert "declares no accumulator" in str(refusal.value)
