"""`scatter_by_index`: the declared indexed write, and the contract it declares.

The catalogue survey ranked `indexed_write` first (23 tasks). Three semantics are
fixed here rather than discovered later: out-of-bounds entries are discarded
(symmetric with `gather_rows` returning zero), duplicates are last-write-wins and
unordered, and the lowering is two launches because a pure operator cannot rewrite
its base.
"""
from types import SimpleNamespace

import pytest

from flowgpu.compiler.analysis.access_shape import SCATTER, plan_shape, platform_conflicts
from flowgpu.compiler.backend.platform_configs import PLATFORMS
from flowgpu.compiler.backend.reference import inputs_for, run as run_reference
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.frontend.diagnostic import CompileError

SOURCE = ("pipeline scatter {\n"
          " input base:fp32[8,4] input index:int32[3] input values:fp32[3,4]\n"
          " out = scatter_by_index(base, index, values)\n"
          " output out\n"
          "}\n")


def compile_program(source=SOURCE):
    return compile_source(source).pipelines[0]


def test_the_output_keeps_the_destination_shape():
    (name, tensor_type) = compile_program().reference.outputs[0]
    assert name == "out"
    assert tensor_type.shape == ["8", "4"]
    assert tensor_type.dtype == "fp32"


def test_the_lowering_is_a_copy_then_an_indexed_write():
    kernels = list(compile_program().triton.kernels)
    assert [kernel.name.rsplit("_", 1)[-1] for kernel in kernels] == ["copy", "write"]
    copy, write = kernels
    assert "tl.store(out_ptr + e" in copy.source.replace("_ptr", "_ptr") or "tl.store" in copy.source
    # the write kernel masks on the index being inside the destination
    assert "target >= 0" in write.source and "ROWS" in write.source
    assert write.grid.startswith("(")


def test_an_index_vector_of_the_wrong_length_is_refused():
    source = SOURCE.replace("input index:int32[3]", "input index:int32[5]")
    with pytest.raises(CompileError, match='scatter_by_index needs one index per value row') as caught:
        compile_program(source)
    assert "one index per value row" in str(caught.value)


def test_a_symbolic_index_extent_is_refused_either_by_the_row_check_or_the_literal_rule():
    """A symbolic index extent fails the "one index per value row" check first,
    because the value rows are literal; the literal rule backs it up."""
    source = SOURCE.replace("input index:int32[3]", "input index:int32[M]")
    with pytest.raises(CompileError, match='scatter_by_index needs one index per value row') as caught:
        compile_program(source)
    message = str(caught.value)
    assert "one index per value row" in message or "literal" in message


def test_the_reference_discards_out_of_bounds_and_copies_the_rest():
    pipeline = compile_program()
    reference = pipeline.reference
    produced = run_reference(reference)["out"]
    base, index, values = [tensor for _, _, _, tensor in inputs_for(reference, {})]
    expected = list(base)
    for row in range(3):
        target = index[row]
        if 0 <= target < 8:
            for offset in range(4):
                expected[target * 4 + offset] = values[row * 4 + offset]
    assert produced == pytest.approx(expected)


def test_the_ir_calls_this_a_scatter_and_refuses_it_on_kunlunxin():
    plan = [SimpleNamespace(ops=("scatter_by_index",))]
    assert plan_shape(plan).kind == SCATTER
    assert len(platform_conflicts(plan, PLATFORMS["kunlunxin"])) == 1
