import pytest

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.lowering_ir import (
    AutoReduce,
    IndexSpace,
    LayoutMap,
    LayoutOp,
    Loop,
    Range,
)


def test_layout_ir_dataclasses_validate_shape_and_mapping():
    space = IndexSpace(("N", "64"))
    mapping = LayoutMap("reshape", (("N", "64"),), ("N", "8", "8"))
    op = LayoutOp("reshape", ("x",), "y", space, mapping)
    assert op.mapping.kind == "reshape"
    assert op.index_space.shape == ("N", "64")


def test_layout_ir_rejects_split_without_part():
    with pytest.raises(ValueError, match="part"):
        LayoutMap("split", (("N", "64"),), ("N", "32"))


def test_loop_and_autoreduce_capture_two_sweep_contract():
    chunk = Range("column", "0", "min(128, next_power_of_2(D))")
    reduction = AutoReduce("sum_last", "x", "y", -1, "fp32", chunk,
                           ("fold", "finalize"), "identity")
    loop = Loop((Range("row", "0", "N"),), (reduction,))
    assert loop.body[0].stages == ("fold", "finalize")
    assert loop.ranges[0].stop == "N"


def test_pipeline_exposes_structured_layout_description_without_changing_emission():
    source = """pipeline p {
        input x : bf16[N, 64]
        a, b = split(x)
        y = concat(a, b)
        output y
    }"""
    result = compile_source(source).pipelines[0]
    assert [item.kind for item in result.lowering_ir] == ["layout", "layout", "layout"]
    assert result.lowering_ir[0].body.mapping.part == 0
    assert result.lowering_ir[1].body.mapping.part == 1
    assert result.triton.complete


def test_pipeline_exposes_autoreduce_description():
    result = compile_source("""pipeline p {
        input x : fp32[N, D]
        y = sum_last(x)
        output y
    }""").pipelines[0]
    reduction = next(item for item in result.lowering_ir if item.kind == "reduction")
    assert isinstance(reduction.body, Loop)
    assert isinstance(reduction.body.body[0], AutoReduce)
    assert reduction.body.body[0].accumulator_dtype == "fp32"
