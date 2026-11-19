"""21.7's stability check.

Two halves, and they moved on 2026-10-20.

Until v0.5 added the elementary value layer, *no* expansion could be written:
`rsqrt`, `sum` and `div` were all unregistered, so the check was provably empty
on every real program and the only thing worth asserting was that boundary.
`rsqrt` now exists, which makes the rms_norm expansion writable as
`mul -> sum_last -> rsqrt` -- so that half of the check is live without any mock
registration, and the boundary has shrunk to the single name `sum`.

What survives from the old file: the boundary is still asserted numerically
(it changes when 19 grows, deliberately), and the mock registration of `sum`
still exists to exercise the patterns that remain unwritable. What is new: the
rms_norm candidate is now produced by a real program, which is the first time
this check has fired without help.
"""

import json

import pytest

from compiler.cli import main
from compiler.compile import compile_source
from compiler.diagnostic import CompileError
from compiler.operator import OPERATORS, infer_mul, infer_unary_same_type, register_operator
from compiler.stability import PATTERNS, boundary, check

HAND_WRITTEN_SOFTMAX = """
pipeline hand_softmax {
    input x : bf16[N,D]
    e = exp(x)
    s = sum(e)
    r = div(e, s)
    output r
}
"""

# The version that needs the mock: `sum` does not exist.
HAND_WRITTEN_RMS_NORM = """
pipeline hand_rms_norm {
    input x : bf16[N,D]
    squared = mul(x, x)
    total = sum(squared)
    scale = rsqrt(total)
    output scale
}
"""

# The version that compiles today, with no mock and no missing operator. This is
# the expansion 21.7 warns about, written in the operators the language has.
HAND_WRITTEN_RMS_NORM_TODAY = """
pipeline hand_rms_norm_today {
    input x : bf16[N,D]
    squared = mul(x, x)
    total = sum_last(squared)
    scale = rsqrt(total)
    output scale
}
"""

USES_THE_PRIMITIVE = """
pipeline uses_softmax {
    input x : bf16[N,D]
    y = softmax(x)
    output y
}
"""

UNCONNECTED = """
pipeline unconnected {
    input x : bf16[N,D]
    input y : bf16[N,D]
    input z : bf16[N,D]
    squared = mul(x, x)
    total = sum(y)
    scale = rsqrt(z)
    output scale
}
"""


@pytest.fixture
def escape_hatch():
    """Register the operators 19 lacks, so that the check has something to see.

    This is exactly what P6 would add. The rules are the simplest ones that make
    the programs above typecheck: `sum` is registered shape-preserving, which a
    real reduction would not be, because 21.7's check reads operators and edges
    and never shapes.
    """
    saved = dict(OPERATORS)

    # Only `sum` is missing from 19 now; `exp`, `div` and `rsqrt` are real
    # operators as of v0.5, and shadowing them with looser rules would make the
    # mock test stop being about the language.
    register_operator("sum", 1, infer_unary_same_type, pointwise=False, acc="fp32")

    yield

    OPERATORS.clear()
    OPERATORS.update(saved)


def candidates_of(source):
    return check(compile_source(source).pipelines[0].dataflow)


@pytest.mark.parametrize(
    "path",
    ["examples/residual_norm.fgpu", "examples/mlp.fgpu", "examples/qk_transform.fgpu"],
)
def test_every_example_is_clean(path):
    assert candidates_of(open(path).read()) == ()


def test_the_boundary_is_exactly_the_operators_the_language_lacks():
    """This test is meant to fail when 19 grows, deliberately.

    21.7 says the check holds today only because the language cannot express the
    expansions, and that its present value is pinning that boundary. A boundary
    nothing asserts is not pinned, so the four names are asserted rather than
    merely printed.
    """
    assert boundary() == ("sum",)
    assert not any(name in OPERATORS for name in boundary())


def test_every_pattern_is_about_a_primitive_the_language_owns():
    for pattern in PATTERNS:
        assert pattern.primitive in OPERATORS
        # Nothing is expanded out of another primitive: the witness operators
        # are the ones a program could write today, if they existed.
        assert pattern.primitive not in pattern.steps


def test_the_softmax_expansion_still_cannot_be_written():
    """The same claim as before, now true of one pattern instead of all of them.

    The softmax expansion needs `sum`, which 19 does not have, so the program is
    refused long before stability runs -- the check cannot silently miss a
    hand-written softmax by failing to match one.
    """
    assert {"exp", "div", "sum_last"} <= set(OPERATORS)
    assert "sum" not in OPERATORS

    with pytest.raises(CompileError, match="olumn 9: Unknown operator 'sum'") as refusal:
        compile_source(HAND_WRITTEN_SOFTMAX)

    assert "sum" in str(refusal.value)


def test_the_rms_norm_expansion_is_now_writable_and_reported():
    """v0.5's consequence, asserted rather than assumed.

    `rsqrt` entering the language makes 21.7's own rms_norm example expressible,
    and the point of the check is that it then fires. This is the first candidate
    in the file produced without a mock registration.
    """
    assert {"mul", "sum_last", "rsqrt"} <= set(OPERATORS)

    candidates = candidates_of(HAND_WRITTEN_RMS_NORM_TODAY)

    assert [candidate.primitive for candidate in candidates] == ["rms_norm"]
    assert candidates[0].witness == ("mul", "sum_last", "rsqrt")
    assert "hand-unrolled" in candidates[0].message()


def test_a_hand_written_softmax_is_reported(escape_hatch):
    candidates = candidates_of(HAND_WRITTEN_SOFTMAX)

    assert len(candidates) == 1
    assert candidates[0].primitive == "softmax"
    assert candidates[0].witness == ("exp", "sum", "div")
    assert candidates[0].values == ("e", "s", "r")
    assert "hand-unrolled" in candidates[0].message()
    assert "21.5" in candidates[0].message()


def test_a_hand_written_rms_norm_is_reported(escape_hatch):
    candidates = candidates_of(HAND_WRITTEN_RMS_NORM)

    assert [candidate.primitive for candidate in candidates] == ["rms_norm"]
    assert candidates[0].witness == ("mul", "sum", "rsqrt")


def test_calling_the_primitive_is_not_a_candidate(escape_hatch):
    assert candidates_of(USES_THE_PRIMITIVE) == ()


def test_witnesses_have_to_be_connected(escape_hatch):
    # mul, sum and rsqrt all occur, but `sum` does not consume what `mul`
    # produced: three unrelated operators are not a hand-written rms_norm.
    assert candidates_of(UNCONNECTED) == ()


def test_a_candidate_does_not_stop_compilation(escape_hatch, capsys, tmp_path):
    assert main(["examples/mlp.fgpu", "--emit", "stability"]) == 0
    assert "candidates  none" in capsys.readouterr().out

    path = tmp_path / "hand_softmax.fgpu"
    path.write_text(HAND_WRITTEN_SOFTMAX)

    # A candidate is advice about a program that compiles, so the compiler
    # reports it and exits 0 rather than refusing the program as an error would.
    assert main([str(path), "--emit", "stability"]) == 0

    printed = capsys.readouterr().out
    assert "candidate   softmax looks hand-unrolled" in printed
    assert "Use softmax: its numerical behaviour" in printed


def test_the_view_states_the_empty_case_and_the_boundary(capsys):
    main(["examples/residual_norm.fgpu", "--emit", "stability"])
    printed = capsys.readouterr().out

    assert "candidates  none" in printed
    assert "rms_norm     mul -> sum -> rsqrt" in printed
    assert "19 registers no sum" in printed


def test_the_json_carries_the_candidates_and_the_boundary(capsys):
    main(["examples/mlp.fgpu", "--emit", "json"])
    stability = json.loads(capsys.readouterr().out)["pipelines"][0]["stability"]

    assert stability["candidates"] == []
    assert stability["boundary"] == ["sum"]
    assert stability["patterns"] == [
        {"primitive": "rms_norm", "steps": ["mul", "sum", "rsqrt"]},
        {"primitive": "rms_norm", "steps": ["mul", "sum_last", "rsqrt"]},
        {"primitive": "softmax", "steps": ["exp", "sum", "div"]},
        {"primitive": "softmax", "steps": ["exp", "sum_last", "div"]},
    ]


def test_real_last_axis_expansion_is_inspected_without_registering_mock_operators():
    src = """pipeline p {
        input x : fp32[N,D]
        e = exp(x)
        s = sum_last(e)
        denominator = broadcast_like(s,e)
        y = div(e,denominator)
        output y
    }"""
    candidates = candidates_of(src)
    assert len(candidates) == 1
    assert candidates[0].witness == ("exp", "sum_last", "div")
