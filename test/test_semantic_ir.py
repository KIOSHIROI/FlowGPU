"""Intent preservation and permutation rewrites checked against independent math."""
import json
from dataclasses import replace

import pytest

from compiler.compile import compile_source
from compiler import reference
from compiler.cache import cached_module
from compiler.cli import main
from compiler.diagnostic import CompileError, DiagnosticKind
from compiler.semantic_ir import normalize, OperationAttributes
from compiler.shape_constraints import equal


def compiled(source, folding=True):
    return compile_source(source, semantic_view_folding=folding,
                          index_fusion=False, reduction_fusion=False).pipelines[0]


def execute(p, *args):
    ns = {}
    exec(reference.module([p.reference]), ns)
    return ns['run_p'](*args)


@pytest.mark.parametrize('op', ['sum_axis', 'max_axis', 'softmax_axis', 'rms_norm_axis'])
@pytest.mark.parametrize('axis', [0, 1, 2, -1, -2, -3])
def test_axes_survive_frontend_until_normalization(op, axis):
    weight = f'input w : fp32[{(2,3,4)[axis%3]}]' if op == 'rms_norm_axis' else ''
    args = 'x,w' if weight else 'x'
    p = compiled(f'pipeline p {{ input x : fp32[2,3,4] {weight} y={op}({args},{axis}) output y }}')
    axis_node = next(n for n in p.semantic.nodes if n.op == op)
    assert axis_node.attributes.axis == axis % 3
    assert axis_node.inputs == (('x','w') if weight else ('x',))
    assert not any(n.op == 'move_first_last' for n in p.semantic.nodes)
    assert not any(n.op.endswith('_axis') for n in p.dataflow.nodes)
    assert any(d.value == 'y' and d.action == 'lower_static_axis' for d in p.normalization)


def test_axis_view_composition_removes_actual_work_without_changing_math():
    src = 'pipeline p { input x : fp32[2,3,4] t=move_first_last(x) y=sum_axis(t,1) output y }'
    old, new = compiled(src, False), compiled(src)
    x = list(range(24))
    # Output [3,1,2]; sum over original final axis, independently indexed.
    want = [sum(x[(b*3+d)*4+k] for k in range(4)) for d in range(3) for b in range(2)]
    assert execute(old, x)['y'] == execute(new, x)['y'] == want
    assert len(old.triton.kernels) == 5 and len(new.triton.kernels) == 2
    assert len(new.memory.slots) < len(old.memory.slots)
    assert any(d.action == 'cancel_identity_view' for d in new.normalization)
    assert new.semantic.as_dict() == old.semantic.as_dict()


def test_repeated_dimension_sizes_do_not_prove_identity():
    p = compiled('pipeline p { input x : fp32[2,2] t=transpose(x) y=relu(t) output y }')
    assert execute(p, [1,2,3,4])['y'] == [1,3,2,4]
    assert any(n.op == 'transpose' for n in p.dataflow.nodes)
    assert not any(d.action == 'cancel_identity_view' for d in p.normalization)


def test_internal_inverse_transposes_and_observable_intermediate():
    src = 'pipeline p { input x : fp32[2,3] t=transpose(x) u=transpose(t) y=relu(u) output t output y }'
    p = compiled(src)
    assert execute(p, [-1,2,-3,4,-5,6]) == {'t':[-1,4,2,-5,-3,6], 'y':[0,2,0,4,0,6]}
    assert len([n for n in p.dataflow.nodes if n.op == 'transpose']) == 1


def test_exported_identity_copy_keeps_existing_storage_boundary():
    p = compiled('pipeline p { input x : fp32[2,3] t=transpose(x) u=transpose(t) a=u output a }')
    assert len([n for n in p.dataflow.nodes if n.op == 'transpose']) == 2
    assert execute(p, [1,2,3,4,5,6])['a'] == [1,2,3,4,5,6]
    assert p.dataflow.resolve('a') == 'u'


def test_cast_boundary_is_not_crossed_by_view_folding():
    p = compiled('pipeline p { input x : fp32[2,2] input h : fp16[2,2] t=transpose(x) c=cast_like(t,h) u=transpose(c) y=to_fp32(u) output y }')
    assert len([n for n in p.dataflow.nodes if n.op == 'transpose']) == 2
    import struct
    x = [0.1,0.3,-0.7,1.0001]
    want = [struct.unpack('e', struct.pack('e', v))[0] for v in x]
    assert execute(p, x, [0]*4)['y'] == want


def test_reshape_obligations_and_shape_constraints_remain_explicit():
    src = 'pipeline p { input x : fp32[N] input other : fp32[M] t=reshape(x,[M]) y=reshape(t,[N]) output y }'
    p = compile_source(src, shape_constraints=[equal('N','M')], index_fusion=False).pipelines[0]
    assert next(n for n in p.semantic.nodes if n.op == 'reshape').attributes.target_shape == ('M',)
    assert p.semantic.as_dict()['shape_constraints'] == [{'operator':'equal','left':'N','right':'M'}]
    assert len([n for n in p.dataflow.nodes if n.op == 'reshape']) == 2
    with pytest.raises(ValueError, match='reshape does not conserve elements'):
        execute(p, [1,2,3,4], [0,0,0], 4, 3)


@pytest.mark.parametrize('attrs', [OperationAttributes(), OperationAttributes(axis=3),
                                  OperationAttributes(axis=True)])
def test_corrupted_axis_attributes_report_internal_diagnostic(attrs):
    p = compiled('pipeline p { input x : fp32[2,3] y=sum_axis(x,0) output y }')
    nodes = tuple(replace(n, attributes=attrs) if n.op == 'sum_axis' else n for n in p.semantic.nodes)
    with pytest.raises(CompileError, match="ipeline 'p': invalid semantic IR: sum_axis needs a normalized static axis") as err:
        replace(p.semantic, nodes=nodes)
    assert err.value.kind == DiagnosticKind.INTERNAL


def test_semantic_ir_is_not_mutated_by_normalization():
    p = compiled('pipeline p { input x : fp32[2,3,4] t=move_first_last(x) y=max_axis(t,1) output y }')
    before = p.semantic.as_dict()
    normalize(p.semantic, fold_views=False)
    normalize(p.semantic, fold_views=True)
    assert p.semantic.as_dict() == before


def test_cli_and_cache_expose_the_actual_policy(tmp_path, capsys):
    src = 'pipeline p { input x : fp32[2,3,4] t=move_first_last(x) y=sum_axis(t,1) output y }'
    path = tmp_path/'p.fgpu'; path.write_text(src)
    assert main([str(path), '--emit', 'semantic']) == 0
    payload = json.loads(capsys.readouterr().out)[0]
    assert next(n for n in payload['nodes'] if n['op'] == 'sum_axis')['axis'] == 1
    assert any(d['action'] == 'cancel_identity_view' for d in payload['normalization'])
    assert main([str(path), '--emit', 'json', '--no-view-folding']) == 0
    payload = json.loads(capsys.readouterr().out)['pipelines'][0]
    assert not any(d['action'] == 'cancel_identity_view' for d in payload['normalization'])
    opts = dict(index_fusion=False, reduction_fusion=False, cache_dir=tmp_path/'cache')
    a, hit_a, key_a = cached_module(src, semantic_view_folding=True, **opts)
    b, hit_b, key_b = cached_module(src, semantic_view_folding=False, **opts)
    assert key_a != key_b and a != b and not hit_a and not hit_b
    assert cached_module(src, semantic_view_folding=True, **opts)[1]
