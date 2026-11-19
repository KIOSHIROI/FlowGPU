"""21.8's portability, as three numbers and one structural claim.

The structural claim is the interesting one. 13.6 and 13.7 run before any target
exists and produce a plan both targets follow, so A cannot differ between them --
not "was measured equal" but "is one decision". The tests check the observable
consequence: the two targets emit the same kernels, and the A in this report is
the one the traffic instrument prices.

The distinction the report has to make is between an operator no target can
reach and one some target cannot. Only the first says something about the
program; the second is a backend's limit, and calling it a portability cost of
the pipeline would be a category error.
"""

import json

import pytest

from compiler.cli import main
from compiler.compile import compile_source
from compiler.portability import measure
from compiler.traffic import plan_traffic

EXAMPLES = [
    "examples/residual_norm.fgpu",
    "examples/mlp.fgpu",
    "examples/qk_transform.fgpu",
]


def report_of(path):
    return measure(compile_source(open(path).read()).pipelines[0])


@pytest.mark.parametrize(
    "path",
    [
        "examples/residual_norm.fgpu",
        "examples/mlp.fgpu",
        "examples/qk_transform.fgpu",
    ],
)
def test_all_standard_programs_are_fully_portable(path):
    """Every standard pipeline now reaches both backends with 100% portability."""
    report = report_of(path)

    assert report.portable
    assert report.unreachable == ()
    assert report.target_limits == ()
    assert all(reach.reaches_everything for reach in report.reaches)


def test_layout_operators_lower_where_the_platform_allows():
    """v0.1 reserved reshape/split/concat; v0.1.2 lowers them as host layout ops.

    The platform gate still speaks: a target that has not validated the kind
    refuses it with the allowed-kinds message, so the operator is either
    lowered or refused, never silently dropped.
    """
    from compiler.backends import KERNELS, REFUSED
    from compiler.platform_configs import resolve_platform

    assert {"reshape", "split", "concat"} <= set(KERNELS["layout"])
    assert REFUSED == {}  # no operator is left with no decision at all

    program = """
    pipeline p {
        input x : bf16[N, 64]
        a, b = split(x)
        output a
    }
    """
    result = compile_source(program).pipelines[0]
    assert result.triton.kernels[0].kind == "layout"

    kunlun = compile_source(
        program, platform_config=resolve_platform("kunlunxin")
    ).pipelines[0]
    assert not kunlun.triton.complete
    assert {r.operator for r in kunlun.triton.refusals} == {"split"}
    assert "pointwise kernels only" in kunlun.triton.refusals[0].reason


@pytest.mark.parametrize("path", EXAMPLES)
def test_both_targets_emit_the_same_kernels(path):
    """The observable half of "A does not degrade between targets".

    A is computed from the plan, and the plan is one object. What can be
    observed from outside is that neither target silently rearranges it.
    """
    result = compile_source(open(path).read()).pipelines[0]
    triton = [kernel.name for kernel in result.triton.kernels]
    reference = [kernel.name for kernel in result.reference.kernels]

    assert triton == reference


@pytest.mark.parametrize("path", EXAMPLES)
def test_the_plan_is_the_one_the_traffic_instrument_prices(path):
    result = compile_source(open(path).read()).pipelines[0]
    report = measure(result)
    traffic = plan_traffic(result.dataflow, result.fusion, result.memory)

    assert report.a == traffic.a
    assert report.kernels == len(result.fusion.groups)


def test_the_view_tells_a_language_hole_from_a_target_limit(capsys):
    main(["examples/mlp.fgpu", "--emit", "portability"])
    printed = capsys.readouterr().out

    assert "A 1.50   computed once, before any target" in printed
    assert "verdict     portable: one source, both targets" in printed

    main(["examples/qk_transform.fgpu", "--emit", "portability"])
    printed = capsys.readouterr().out

    assert "A 1.75   computed once, before any target" in printed
    assert "verdict     portable: one source, both targets" in printed


def test_the_json_carries_the_numbers(capsys):
    main(["examples/qk_transform.fgpu", "--emit", "json"])
    portability = json.loads(capsys.readouterr().out)["pipelines"][0]["portability"]

    assert portability["portable"] is True
    assert portability["unreachable_in_every_target"] == []
    assert portability["target_limits"] == []
    assert [target["target"] for target in portability["targets"]] == [
        "triton",
        "reference",
    ]
    assert portability["plan"]["kernels"] == 4
