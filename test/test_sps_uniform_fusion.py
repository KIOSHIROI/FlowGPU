"""SPS S3 (program-uniform placement) and S4 (fusion-aware cost)."""
from __future__ import annotations

from types import SimpleNamespace

from flowgpu.compiler.compile import compile_pipelines
from flowgpu.compiler.core.program_scalar import register, unregister
from flowgpu.compiler.frontends.python_edsl import program
from flowgpu.compiler.optimization.scalar_placement import Placement, plan_placements


def _scalar_pipeline():
    with program("p") as builder:
        x = builder.input("x", "fp32[1024]")
        c = builder.input("c", "fp32")
        builder.output(x * c)
    return builder.pipelines[0]


class _FakeFusion:
    def __init__(self, groups):
        self.groups = groups


class _FakeGroup:
    def __init__(self, operands):
        self.members = (SimpleNamespace(inputs=list(operands)),)


def test_s3_program_uniform_scalar_appears_as_uniform_in_the_plan():
    register()
    try:
        with program("p") as builder:
            table = builder.input("table", "int32[S]")
            x = builder.input("x", "int32[1024]")
            scalar = builder.call("program_scalar", table)
            builder.output(builder.add(x, scalar))
        result = compile_pipelines(builder.pipelines, scalar_placement="sps")
        pipeline = result.pipelines[0]
        plan = pipeline.scalar_placement_plan
        uniform = [decision for decision in plan.decisions
                   if decision.placement is Placement.UNIFORM]
        assert len(uniform) == 1
        assert uniform[0].name == "t1"
        assert pipeline.triton.complete
        assert pipeline.triton.kernels[0].ops == ("program_scalar", "add")
        # one uniform load, at the kernel top level
        scalar_line = next(line for line in pipeline.triton.kernels[0].source.splitlines()
                           if "= tl.load(table_ptr" in line)
        assert "tl.arange" not in scalar_line and "mask=" not in scalar_line
    finally:
        unregister()


def test_s4_fusion_groups_change_the_amortisation_threshold():
    pipeline = _scalar_pipeline()
    graph = SimpleNamespace(nodes=(), values={})
    calls = 60_000
    one_group = plan_placements(pipeline, graph, specialization={"c": 2.0},
                                calls=calls,
                                fusion=_FakeFusion([_FakeGroup(["c"])]))
    two_groups = plan_placements(pipeline, graph, specialization={"c": 2.0},
                                 calls=calls,
                                 fusion=_FakeFusion([_FakeGroup(["c"]),
                                                     _FakeGroup(["c"])]))
    assert one_group.decisions[0].placement is Placement.HOST
    assert two_groups.decisions[0].placement is Placement.CONSTEXPR
    assert "2 group(s)" in two_groups.decisions[0].reason


def test_s4_real_compile_plan_reports_fusion_group_count():
    with program("p") as builder:
        x = builder.input("x", "fp32[1024]")
        c = builder.input("c", "fp32")
        y = x * c
        builder.output(builder.relu(y))
    result = compile_pipelines(builder.pipelines, scalar_placement="sps",
                               scalar_specialization={"c": 2.0},
                               scalar_calls=10_000_000)
    plan = result.pipelines[0].scalar_placement_plan
    assert plan.decisions[0].placement is Placement.CONSTEXPR
    assert "group(s)" in plan.decisions[0].reason
