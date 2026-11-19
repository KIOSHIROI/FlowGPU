"""SPS S2: the same plan is derived for DSL, python eDSL and imported Triton."""
from __future__ import annotations

from flowgpu.compiler import compile_source
from flowgpu.compiler.compile import compile_pipelines
from flowgpu.compiler.frontends.python_edsl import program
from flowgpu.compiler.frontends.triton_importer import import_triton

DSL = "pipeline p { input x:fp32[1024] input c:fp32 y=mul(x,c) output y }"
TRITON = '''
@triton.jit
def scale_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 2.0, mask=mask)
'''


def _edesl_pipeline():
    with program("p") as p:
        x = p.input("x", "fp32[1024]")
        c = p.input("c", "fp32")
        p.output(x * c)
    return p.pipelines


def _plans():
    dsl = compile_source(DSL, scalar_placement="sps",
                         scalar_specialization={"c": 2.0},
                         scalar_calls=10_000_000).pipelines[0]
    edsl = compile_pipelines(_edesl_pipeline(), scalar_placement="sps",
                             scalar_specialization={"c": 2.0},
                             scalar_calls=10_000_000).pipelines[0]
    imported = import_triton(TRITON, input_types={"x_ptr": "fp32[1024]"},
                             output_types={"out_ptr": "fp32[1024]"})
    triton = compile_pipelines(imported.pipelines, scalar_placement="sps",
                               scalar_specialization=imported.constants,
                               scalar_calls=10_000_000).pipelines[0]
    return dsl, edsl, triton


def test_three_frontends_get_the_same_placement_decision():
    dsl, edsl, triton = _plans()
    for pipeline in (dsl, edsl, triton):
        plan = pipeline.scalar_placement_plan
        assert plan is not None
        assert [d.placement.value for d in plan.decisions] == ["constexpr"]
        assert plan.constexpr_scalars  # one literal per frontend
        assert pipeline.triton.complete
        assert pipeline.triton.host.splitlines()[0].count(",") == 0
    # DSL and eDSL use the same scalar name; the importer names the constant c0
    assert dsl.scalar_placement_plan.constexpr_scalars == {"c": "2.0"}
    assert edsl.scalar_placement_plan.constexpr_scalars == {"c": "2.0"}
    assert triton.scalar_placement_plan.constexpr_scalars == {"c0": "2.0"}


def test_three_frontends_get_the_same_host_decision_for_one_shot_calls():
    dsl = compile_source(DSL, scalar_placement="sps",
                         scalar_specialization={"c": 2.0}, scalar_calls=1).pipelines[0]
    edsl = compile_pipelines(_edesl_pipeline(), scalar_placement="sps",
                             scalar_specialization={"c": 2.0}, scalar_calls=1).pipelines[0]
    for pipeline in (dsl, edsl):
        assert [d.placement.value for d in pipeline.scalar_placement_plan.decisions] == ["host"]
        assert pipeline.scalar_placement_plan.constexpr_scalars == {}
    imported = import_triton(TRITON, input_types={"x_ptr": "fp32[1024]"},
                             output_types={"out_ptr": "fp32[1024]"})
    triton = compile_pipelines(imported.pipelines, scalar_placement="sps",
                               scalar_specialization=imported.constants,
                               scalar_calls=1).pipelines[0]
    assert [d.placement.value for d in triton.scalar_placement_plan.decisions] == ["host"]


def test_three_frontends_compile_the_same_operator_sequence():
    dsl, edsl, triton = _plans()
    sequences = {name: [node.op for node in pipeline.dataflow.nodes
                        if node.op not in ("input", "output")]
                 for name, pipeline in (("dsl", dsl), ("edsl", edsl), ("triton", triton))}
    assert sequences["dsl"] == sequences["edsl"] == sequences["triton"]
    assert sequences["dsl"] == ["mul"]
