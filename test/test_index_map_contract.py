from dataclasses import replace
import pytest

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.ir.index_maps import inverse_coordinates,permutation_of
from flowgpu.compiler.ir.semantic_ir import OperationAttributes
from flowgpu.compiler.frontend.diagnostic import CompileError,DiagnosticKind


def test_axis_lowering_retains_typed_maps_on_generated_rotations():
    p=compile_source('pipeline p { input x : fp32[2,3,4] y=sum_axis(x,0) output y }').pipelines[0]
    views=[n for n in p.dataflow.nodes if n.op=='move_first_last']
    assert views
    for node in views:
        rank=len(p.dataflow.values[node.inputs[0]].shape)
        assert node.attributes is not None
        assert node.attributes.permutation==permutation_of(node,rank)


def test_equal_sized_dimensions_do_not_hide_a_wrong_axis_map():
    p=compile_source('pipeline p { input x : fp32[2,2] y=transpose(x) output y }').pipelines[0]
    node=next(n for n in p.dataflow.nodes if n.op=='transpose')
    assert inverse_coordinates(node,('i','j'))==('j','i')
    p.dataflow.nodes[p.dataflow.index_of(node)]=replace(node,attributes=OperationAttributes(permutation=(0,1)))
    with pytest.raises(CompileError, match="ipeline 'p': invalid dataflow graph: view permutation metadata disagrees with operator/axis identities") as error:p.dataflow.validate()
    assert error.value.kind==DiagnosticKind.INTERNAL


def test_inverse_map_composes_axis_identities_and_legacy_nodes_have_explicit_fallback():
    p=compile_source('pipeline p { input x : fp32[2,3,4] y=move_first_last(x) output y }').pipelines[0]
    node=next(n for n in p.dataflow.nodes if n.op=='move_first_last')
    assert inverse_coordinates(node,('b','c','a'))==('a','b','c')
    assert inverse_coordinates(replace(node,attributes=None),('b','c','a'))==('a','b','c')
    with pytest.raises(ValueError, match='view permutation metadata disagrees with operator/axis identities'):permutation_of(replace(node,attributes=OperationAttributes(permutation=(True,2,0))),3)
