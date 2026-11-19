"""Memory dependences between the kernels of a plan.

The point of these tests is the boundary as much as the analysis: value-level
def-use is free under single assignment, so what is measured here is bytes --
which buffer is written before it is read, and with what certainty.
"""
import pytest

from compiler.analysis.dependences import (
    MAY, MUST, READ, WRITE, AccessSummary, accesses, conflicts, report,
)
from compiler.compile import compile_source


def plan(body, inputs, output="out"):
    source = ("pipeline p {\n" f" input {inputs}\n {body}\n" f" output {output}\n}}\n")
    return list(compile_source(source).pipelines[0].triton.kernels)


def test_a_materialised_result_is_a_true_dependence():
    """Two kernels, one buffer between them: that is RAW, and it must hold."""
    kernels = plan("centered = mul(x, x)\n"
                   "out = patchify(centered, 2, 2)", "x:bf16[1,4,8,8]")
    found = conflicts(kernels)
    raw = [item for item in found if item.kind == "RAW"]
    assert len(raw) == 1
    assert raw[0].certainty == MUST
    assert raw[0].earlier == kernels[0].name and raw[0].later == kernels[1].name


def test_a_fused_plan_has_no_memory_dependence():
    """One kernel, so nothing is materialised between kernels: the plan is
    dependence-free through memory, which is what fusion buys."""
    kernels = plan("out = silu(x)", "x:bf16[4,8]")
    assert len(kernels) == 1
    assert conflicts(kernels) == []


def test_every_access_states_a_role_and_a_certainty():
    kernels = plan("centered = mul(x, x)\n"
                   "out = patchify(centered, 2, 2)", "x:bf16[1,4,8,8]")
    summaries = accesses(kernels)
    assert {item.role for item in summaries} == {READ, WRITE}
    assert {item.certainty for item in summaries} == {MUST}


def test_an_unknown_role_or_certainty_is_rejected():
    with pytest.raises(ValueError, match='unknown access role: sideways'):
        AccessSummary("k", "v", "sideways", "block")
    with pytest.raises(ValueError, match='unknown certainty: probably'):
        AccessSummary("k", "v", READ, "block", "probably")


def test_the_report_is_serialisable_and_names_the_boundary():
    kernels = plan("centered = mul(x, x)\n"
                   "out = patchify(centered, 2, 2)", "x:bf16[1,4,8,8]")
    data = report(kernels)
    assert data["kernels"] == [kernel.name for kernel in kernels]
    assert data["conflicts"] and data["conflicts"][0]["kind"] == "RAW"
    # the module's docstring states what it does not analyse; pin that claim
    from compiler.analysis import dependences
    assert "loop-carried" in dependences.__doc__
