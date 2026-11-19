"""S1: IR-native scalar placement (host vs constexpr) on core pipelines."""
from __future__ import annotations

import pytest

from flowgpu.compiler.compile import compile_pipelines
from flowgpu.compiler.frontends.python_edsl import program
from flowgpu.compiler.optimization.scalar_placement import (Placement, PlacementCostModel,
                                                    plan_placements)


def _scalar_pipeline():
    with program("p") as builder:
        x = builder.input("x", "fp32[1024]")
        c = builder.input("c", "fp32")
        builder.output(x * c)
    return builder.pipelines


def _compile(**options):
    return compile_pipelines(_scalar_pipeline(), **options).pipelines[0]


def test_plan_keeps_host_for_a_one_shot_call():
    pipeline = _scalar_pipeline()[0]
    plan = plan_placements(pipeline, None, specialization={"c": 2.0}, calls=1)
    assert plan.decisions[0].placement is Placement.HOST
    assert plan.constexpr_scalars == {}


def test_plan_specialises_when_the_compile_amortises():
    pipeline = _scalar_pipeline()[0]
    plan = plan_placements(pipeline, None, specialization={"c": 2.0}, calls=10_000_000)
    assert plan.decisions[0].placement is Placement.CONSTEXPR
    assert plan.constexpr_scalars == {"c": "2.0"}
    assert plan.decisions[0].literal == "2.0"


def test_host_plan_keeps_the_host_abi():
    pipeline = _compile(scalar_placement="sps", scalar_specialization={"c": 2.0},
                        scalar_calls=1)
    assert pipeline.scalar_placement_plan is not None
    assert pipeline.triton.host.splitlines()[0] == "def run_p(x, c):"
    assert "2.0" not in pipeline.triton.kernels[0].source


def test_constexpr_plan_removes_the_host_abi_and_inlines_the_literal():
    pipeline = _compile(scalar_placement="sps", scalar_specialization={"c": 2.0},
                        scalar_calls=10_000_000)
    assert pipeline.scalar_placement_plan.decisions[0].placement is Placement.CONSTEXPR
    assert pipeline.triton.host.splitlines()[0] == "def run_p(x):"
    assert "2.0" in pipeline.triton.kernels[0].source
    # the host no longer reads the specialised scalar
    assert "input c is a scalar" not in pipeline.triton.host


def test_default_off_is_unchanged():
    pipeline = _compile()
    assert pipeline.scalar_placement_plan is None
    assert pipeline.triton.host.splitlines()[0] == "def run_p(x, c):"


def test_bad_specialisation_names_and_options_are_rejected():
    pipeline = _scalar_pipeline()
    with pytest.raises(ValueError):
        compile_pipelines(pipeline, scalar_placement="nonsense")
    with pytest.raises(ValueError):
        compile_pipelines(pipeline, scalar_placement="sps",
                          scalar_specialization={"missing": 1.0}, scalar_calls=1)
    with pytest.raises(ValueError):
        compile_pipelines(pipeline, scalar_placement="sps", scalar_calls=0)
    with pytest.raises(TypeError):
        compile_pipelines(pipeline, scalar_placement="sps", scalar_specialization=[])
    with pytest.raises(ValueError):
        plan_placements(pipeline[0], None, specialization={"c": "not-a-number"})


def test_imported_triton_constants_can_be_specialised():
    from flowgpu.compiler.frontends.triton_importer import import_triton
    source = '''
@triton.jit
def scale_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)
'''
    imported = import_triton(source, input_types={"x_ptr": "fp32[1024]"},
                             output_types={"out_ptr": "fp32[1024]"})
    pipeline = compile_pipelines(
        imported.pipelines, scalar_placement="sps",
        scalar_specialization=imported.constants, scalar_calls=10_000_000).pipelines[0]
    assert set(pipeline.scalar_placement_plan.constexpr_scalars) == {"c0"}
    assert pipeline.triton.host.splitlines()[0] == "def run_scale_kernel(x_ptr):"
    assert "2.0" in pipeline.triton.kernels[0].source
