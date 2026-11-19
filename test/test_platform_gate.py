"""The pre-emission platform gate, and the two contracts it sits beside.

Before this, `analysis/access_shape.platform_conflicts` was a function nobody
called during compilation: the only way to learn that a target was measured not
to sustain a plan's addressing was `--emit triton` and then `contracts.violations`
on the text. That is an *after-the-fact* checker: by the time it can answer, a
kernel has been written and printed. The fusion report is the last point where
the plan is fully decided and no source exists, so the verdict belongs there.

Two rules shape the tests below:

* **Report, then refuse.** The conflict list is computed on every compile and
  carried on `PipelineResult`, so `--emit contracts` and `--emit json` can show
  it without a device and without changing what compiles. Refusing is opt-in
  (`platform_gate=True`), because turning a checker into a refusal changes which
  programs compile and `docs/COMPILER_ACCEPTANCE.md` requires parent/candidate
  evidence for a default change -- `default_changed: False`.
* **The gate is not the whitelist.** On today's platform table 昆仑芯 allows only
  `pointwise`, so every non-pointwise plan is already refused by the kind gate;
  the access gate is redundant there. Its value is that it is a different
  statement -- *measured access shape*, not *permission* -- and it becomes the
  only thing that speaks as soon as a target enables a kind without enabling the
  access. The synthetic platform in
  `test_the_gate_is_load_bearing_when_the_whitelist_says_yes` is that target.
"""
from dataclasses import replace

import pytest

from compiler.analysis.access_shape import (
    BLOCK,
    GATHER,
    PERMUTED,
    SCATTER,
    group_conflicts,
    group_shape,
)
from compiler.backend.platform_configs import PLATFORMS
from compiler.compile import compile_source
from compiler.frontend.diagnostic import CompileError

PROGRAMS = {
    "gather": ("pipeline g {\n input x:fp32[8,4] input i:int32[4]\n"
               " y = gather_rows(x, i)\n output y\n}\n"),
    "permuted": ("pipeline p {\n input x:fp32[2,4,4,4]\n"
                 " y = patchify(x, 2, 2)\n output y\n}\n"),
    "scatter": ("pipeline s {\n input base:fp32[8,4] input index:int32[3]"
                " input values:fp32[3,4]\n out = scatter_by_index(base, index, values)\n"
                " output out\n}\n"),
    "block": ("pipeline b {\n input x:fp32[4,8]\n y = mul(x, x)\n output y\n}\n"),
    "pad_cat": ("pipeline c {\n input x:fp32[4,2,8,8,8] input cache:fp32[4,2,2,8,8]\n"
                " y = pad_cat(x, cache, 0,0,0,0,4,0)\n output y\n}\n"),
}


def compile_for(name, platform="cuda", **options):
    return compile_source(PROGRAMS[name], platform_config=PLATFORMS[platform],
                          **options).pipelines[0]


# --------------------------------------------------------------------------
# the plan-level verdict, before any source exists
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name, expected", [
    ("gather", GATHER),
    ("permuted", PERMUTED),
    ("scatter", SCATTER),
    ("block", BLOCK),
    ("pad_cat", BLOCK),
])
def test_the_plan_shape_is_decided_from_the_fusion_plan(name, expected):
    """`group_conflicts` runs on the fusion report, i.e. before `lower()`.

    That is the whole point: the descriptor is available at the last moment the
    plan is decided, so a target can be told "no" before a line is written.
    """
    assert group_shape(compile_for(name).fusion).kind == expected


def test_the_plan_is_judged_over_every_group_not_only_the_emitted_ones():
    """Why the authoritative form is pre-emission.

    On 昆仑芯 the `patchify` kernel is refused by the kind whitelist, so the
    *emitted* plan is the `mul` kernel alone and its shape reads `block` -- the
    verdict weakens precisely when a kernel is dropped. The fusion plan still
    contains both groups, so `group_shape` reads `permuted` and the gate can
    refuse what the post-hoc view would call legal.
    """
    from compiler.analysis.access_shape import plan_shape

    source = open("examples/batch10/normalize_and_patchify.fgpu", encoding="utf-8").read()
    pipeline = compile_source(source, platform_config=PLATFORMS["kunlunxin"]).pipelines[0]

    assert group_shape(pipeline.fusion).kind == PERMUTED
    assert plan_shape(pipeline.triton.kernels).kind == BLOCK     # the trap
    assert group_conflicts(pipeline.fusion, PLATFORMS["kunlunxin"])


def test_the_gate_reports_nothing_on_a_target_that_sustains_the_access():
    for name in PROGRAMS:
        assert group_conflicts(compile_for(name).fusion, PLATFORMS["cuda"]) == [], name


@pytest.mark.parametrize("name", ["gather", "permuted", "scatter"])
def test_the_gate_names_the_access_shape_and_the_remedy_on_kunlunxin(name):
    """A refusal that does not say *which* shape broke and what to do instead is
    a refusal a user cannot act on."""
    conflicts = group_conflicts(compile_for(name).fusion, PLATFORMS["kunlunxin"])
    assert len(conflicts) == 1
    assert conflicts[0].startswith("kunlunxin:")
    assert any(word in conflicts[0] for word in ("permutation", "gather", "scatter"))
    assert "iterate the" in conflicts[0]


def test_a_block_plan_passes_the_gate_on_kunlunxin():
    """Task144's `pad_cat` measured 5/5 legal there; a gate that flagged every
    plan on the target would be useless."""
    for name in ("block", "pad_cat"):
        assert group_conflicts(compile_for(name).fusion, PLATFORMS["kunlunxin"]) == [], name


# --------------------------------------------------------------------------
# report on every compile, refuse only when asked
# --------------------------------------------------------------------------
def test_the_verdict_is_carried_by_the_result_without_changing_what_compiles():
    """Report-only by default: the conflicts are on the result, and the kernels
    are still emitted. Turning this into a refusal by default would change which
    programs compile, which needs evidence this increment does not have."""
    pipeline = compile_for("permuted", platform="kunlunxin")
    assert pipeline.platform_conflicts, "the verdict must be reported"
    assert pipeline.triton is not None


def test_the_gate_refuses_before_emission_when_asked_for():
    with pytest.raises(CompileError, match="ipeline 'p': this target is measured not to sustain the plan's addressing: kunlunxin: the plan's access is a static element permutation \\(operator patchify permutes elements by a static patch map\\); iterate the source densely and move the permutation into the store address") as caught:
        compile_for("permuted", platform="kunlunxin", platform_gate=True)
    message = str(caught.value)
    assert "not to sustain" in message and "kunlunxin" in message
    assert "permutation" in message


def test_the_gate_still_compiles_what_the_target_sustains():
    assert compile_for("permuted", platform="cuda", platform_gate=True).triton is not None


def test_the_gate_flag_must_be_a_bool():
    with pytest.raises(ValueError, match="platform_gate"):
        compile_for("block", platform_gate="yes")


# --------------------------------------------------------------------------
# the case that makes the gate more than a duplicate of the kind whitelist
# --------------------------------------------------------------------------
MEASURED_BUT_PERMITTED = replace(PLATFORMS["cuda"], name="synthetic-partial",
                                 gather_load_ok=False)


def test_the_gate_is_load_bearing_when_the_whitelist_says_yes():
    """A target that *permits* every kernel kind while being measured not to
    sustain a gather. No shipped platform is in this state, and it is exactly the
    state the two corrections in `RESULTS.md` describe: the permission table said
    yes, the measured access said no.

    On this platform the kind gate is silent, so the access gate is the only
    thing that can refuse -- and it must, because otherwise the plan is emitted
    and only a text audit after the fact finds it.
    """
    source = PROGRAMS["gather"]
    pipeline = compile_source(source, platform_config=MEASURED_BUT_PERMITTED).pipelines[0]

    # the kind whitelist lets it through: a kernel exists
    assert pipeline.triton.kernels
    assert pipeline.platform_conflicts, "the access gate must be the one that speaks"

    with pytest.raises(CompileError, match="not to sustain"):
        compile_source(source, platform_config=MEASURED_BUT_PERMITTED, platform_gate=True)


def test_the_two_contracts_agree_on_the_case_where_both_speak():
    """The IR verdict and the text verdict are derived from different things --
    operator identity vs emitted expressions -- so agreement is evidence and
    disagreement is a bug in one of them."""
    from compiler.backend.contracts import violations

    pipeline = compile_source(PROGRAMS["gather"],
                              platform_config=MEASURED_BUT_PERMITTED).pipelines[0]
    text_problems = [problem
                     for kernel in pipeline.triton.kernels
                     for problem in violations(kernel.source, MEASURED_BUT_PERMITTED)]
    assert pipeline.platform_conflicts and text_problems
    assert any("gather" in problem for problem in text_problems)


# --------------------------------------------------------------------------
# what the gate must NOT claim: the recorded disagreement stays recorded
# --------------------------------------------------------------------------
def test_the_gate_does_not_paper_over_the_task152_disagreement():
    """`RESULTS.md` records that Task152's official 0.03x belongs to the
    hand-written kernel that was submitted, while the compiler emits a `scan`
    whose access is blocked and whose masks are identical. The gate must agree
    with the compiler's own plan -- saying "infeasible" here would erase the
    question the replay left open, and saying it *because* the score was bad
    would be fitting the checker to the answer.
    """
    source = open("examples/batch10/kda_gate_chunk_cumsum.fgpu", encoding="utf-8").read()
    pipeline = compile_source(source, platform_config=PLATFORMS["kunlunxin"]).pipelines[0]
    assert group_shape(pipeline.fusion).kind == BLOCK
    assert group_conflicts(pipeline.fusion, PLATFORMS["kunlunxin"]) == []


def test_the_gate_flags_the_task154_plan():
    """The other half: 154's compiled plan is `patchify`, whose access is a
    permutation, and the IR says so before emission. This is the prediction the
    replay already records as matching the official result."""
    source = open("examples/batch10/normalize_and_patchify.fgpu", encoding="utf-8").read()
    pipeline = compile_source(source, platform_config=PLATFORMS["kunlunxin"]).pipelines[0]
    assert group_shape(pipeline.fusion).kind == PERMUTED
    assert group_conflicts(pipeline.fusion, PLATFORMS["kunlunxin"])


# --------------------------------------------------------------------------
# the CLI surfaces both halves
# --------------------------------------------------------------------------
def test_the_contracts_view_prints_the_plan_and_the_source(capsys):
    from compiler.cli import main

    assert main(["examples/batch10/normalize_and_patchify.fgpu",
                 "--emit", "contracts"]) == 0
    printed = capsys.readouterr().out
    assert "plan (from the IR, before emission)" in printed
    assert "source (from the emitted kernels)" in printed
    assert "permuted" in printed
    assert "agreement" in printed


def test_the_cli_gate_refuses_and_exits_non_zero(tmp_path, capsys):
    from compiler.cli import main

    path = tmp_path / "patched.fgpu"
    path.write_text(PROGRAMS["permuted"])

    assert main([str(path), "--platform", "kunlunxin", "--platform-gate"]) == 1
    assert "not to sustain" in capsys.readouterr().err

    # off by default: the same program still compiles
    assert main([str(path), "--platform", "kunlunxin"]) == 0


def test_the_json_view_carries_the_verdict(capsys):
    import json as _json

    from compiler.cli import main

    main(["examples/batch10/normalize_and_patchify.fgpu", "--platform", "kunlunxin",
          "--emit", "json"])
    pipeline = _json.loads(capsys.readouterr().out)["pipelines"][0]
    assert pipeline["platform_conflicts"]
    assert pipeline["platform_conflicts"][0].startswith("kunlunxin:")

def test_the_contracts_view_separates_partial_from_disagreement(capsys):
    """A refused kernel leaves the text half with nothing to judge, which is not
    the same thing as the two halves contradicting each other. Reporting both as
    "DISAGREE" would train the reader to ignore the line that matters.

    On 昆仑芯 the `patchify` group is refused by the kind whitelist, so the plan
    says `permuted`, the emitted `mul` kernel is clean, and the honest verdict is
    "partial": one group was never written, so it could not be checked twice.
    """
    from compiler.cli import main

    assert main(["examples/batch10/normalize_and_patchify.fgpu",
                 "--platform", "kunlunxin", "--emit", "contracts"]) == 0
    printed = capsys.readouterr().out
    assert "partial" in printed
    assert "DISAGREE" not in printed
