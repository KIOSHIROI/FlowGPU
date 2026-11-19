"""Layout operators accept integers, and what that unblocks.

Measured cause, 2026-10-20: the content census (`classify.py`) kept counting the
MoE post-reorder family (`post_reorder_*`, `deepep_*`) as blocked even after
`gather_rows`/`scatter_by_index` landed. Reading the statements showed why --
those kernels index with a **table**:

    output[t] = sum_i down_output[src2dst[t, i]] * topk_weights[t, i]

and to write that with a row gather, `src2dst` (int32) has to be reshaped from
`[T, K]` to `[T*K]` first. `reshape`/`split`/`concat` refused integer tensors
while `transpose` and `move_first_last` accepted them, which is an
inconsistency rather than a decision: a layout operator moves elements and never
computes them, so the dtype is irrelevant to what it does.

This file pins the relaxation *and* the idiom it unblocks, because a relaxation
with no user is indistinguishable from a loosened rule.
"""
import math

import pytest

from compiler.backend.reference import inputs_for, run as run_reference
from compiler.compile import compile_source
from compiler.frontend.diagnostic import CompileError


def compile_program(source):
    return compile_source(source).pipelines[0]


def feeds(pipeline):
    return {name: values for name, _, _, values in inputs_for(pipeline.reference, {})}


# --------------------------------------------------------------------------
# the relaxation itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("operator, source", [
    ("reshape", "pipeline p {\n input i:int32[4,2]\n o = reshape(i, [8])\n output o\n}\n"),
    ("split", "pipeline p {\n input i:int32[4,4]\n a, b = split(i)\n output a\n}\n"),
    ("concat", "pipeline p {\n input i:int32[4,2]\n o = concat(i, i)\n output o\n}\n"),
])
def test_a_layout_operator_moves_integers_without_complaining(operator, source):
    """Same rule `transpose` always had: moving elements is not arithmetic."""
    pipeline = compile_program(source)
    produced = run_reference(pipeline.reference)
    assert produced


def test_reshape_of_an_index_table_matches_the_flattening_it_claims():
    pipeline = compile_program(
        "pipeline p {\n input i:int32[4,2]\n o = reshape(i, [8])\n output o\n}\n")
    produced = run_reference(pipeline.reference)["o"]
    assert produced == feeds(pipeline)["i"]


def test_layout_still_refuses_booleans():
    """`bool` is not a number (21.6); moving a mask is a layout change, but the
    arithmetic classes are what the operators are stated over."""
    source = "pipeline p {\n input m:bool[4,2]\n o = reshape(m, [8])\n output o\n}\n"
    with pytest.raises(CompileError, match='Cannot reshape a bool') as caught:
        compile_program(source)
    assert "bool is not a number" in str(caught.value)


# --------------------------------------------------------------------------
# what it unblocks: a post-reorder combine, written with the operators 19 has
# --------------------------------------------------------------------------
#: `output[t, :] = sum_i down[src2dst[t, i], :] * weights[t, i]`, for K=2.
#: This is Task101 `post_reorder_deepgemm` / Task65 / Task88, reduced to the
#: smallest literal shape that still exercises the whole chain.
POST_REORDER = """
pipeline post_reorder {
 input down:fp32[8,3]
 input src2dst:int32[4,2]
 input weights:fp32[4,2]
 flat = reshape(src2dst, [8])
 gathered = gather_rows(down, flat)
 blocks = reshape(gathered, [4,2,3])
 scaled = mul(blocks, broadcast_like(reshape(weights, [4,2,1]), blocks))
 total = sum_axis(scaled, 1)
 out = reshape(total, [4,3])
 output out
}
"""


def test_the_post_reorder_combine_compiles_and_runs():
    pipeline = compile_program(POST_REORDER)
    produced = run_reference(pipeline.reference)["out"]
    data = feeds(pipeline)
    down, src2dst, weights = data["down"], data["src2dst"], data["weights"]
    # `gather_rows`' documented contract, replicated exactly: a negative index is
    # normalised once by adding the row count, and a row still out of bounds
    # reads as zero rather than as a refusal.
    expected = []
    for token in range(4):
        acc = [0.0, 0.0, 0.0]
        for slot in range(2):
            row = src2dst[token * 2 + slot]
            if row < 0:
                row += 8
            for column in range(3):
                source = down[row * 3 + column] if 0 <= row < 8 else 0.0
                acc[column] += source * weights[token * 2 + slot]
        expected.extend(acc)
    assert produced == pytest.approx(expected, abs=1e-6)


def test_the_post_reorder_combine_is_a_gather_not_a_scatter():
    """The pre-reorder half writes rows to indexed destinations (a scatter); this
    half only *reads* them out of place, then reduces. The distinction is what
    `analysis/access_shape` reports to a target, so it is asserted rather than
    assumed: an indexed read and an indexed write do not cost the same."""
    from compiler.analysis.access_shape import GATHER, SCATTER, of_op

    assert of_op("gather_rows").kind == GATHER
    assert of_op("scatter_by_index").kind == SCATTER
    # and the reduction that collapses the slot axis is a plain block access
    assert of_op("sum_axis").kind == of_op("sum_last").kind
