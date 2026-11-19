import ast
import json
import math
from pathlib import Path
import pytest
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.triton_backend import module
from flowgpu.compiler import reference
from flowgpu.compiler.cache import cached_module
from flowgpu.compiler.cli import main
from flowgpu.compiler.platform_configs import resolve_platform


def run(src,args,dims=None,**options):
    p=compile_source(src,**options).pipelines[0]
    namespace={};exec(reference.module([p.reference]),namespace)
    return namespace['run_'+p.pipeline.name](*args,*[(dims or {})[d] for d in p.reference.dimensions])


@pytest.mark.parametrize('bindings',[None,{'R':1,'B':1,'H':1,'D':1},{'R':7,'B':2,'H':3,'D':33}])
def test_lse_has_two_kernels_and_exports_cross_domain_weights(bindings):
    src=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()
    # Two kernels and four slots is the materialize partition; naming it keeps
    # this test about that partition now that the default resolves per program.
    p=compile_source(src,shape_bindings=bindings,cross_domain_strategy='materialize').pipelines[0]
    assert len(p.triton.kernels)==2 and len(p.memory.slots)==4
    stats=p.triton.kernels[0]
    assert stats.results==('weights','denom','out_lse')
    assert len([n for n in p.fusion.groups[0].members if n.op in {'max_last','sum_last'}])==2
    assert 'weights' not in p.fusion.fused_values and 'denom' not in p.fusion.fused_values
    assert 'm' in p.fusion.fused_values and 'clean' in p.fusion.fused_values
    members=[p.dataflow.index_of(n) for g in p.fusion.groups for n in g.members]
    assert len(members)==len(set(members))
    for g in p.fusion.groups:
        inside={p.dataflow.index_of(n) for n in g.members}
        for v in g.fused_values:
            assert all(p.dataflow.index_of(c) in inside for c in p.dataflow.consumers_of(v))
        for n in g.members:
            for v in n.inputs:
                at=p.dataflow.index_of(p.dataflow.producer_of(v))
                assert at in inside or at<min(inside)
    ast.parse(module([p.triton]));ast.parse(reference.module([p.reference]))
    assert len(compile_source(src,reduction_fusion=False).pipelines[0].triton.kernels)==6


GENERIC='''pipeline stats {
 input x : fp32[B,R]
 m=max_last(x)
 wide_m=broadcast_like(m,x)
 w=exp(sub(x,wide_m))
 s=sum_last(w)
 wide_s=broadcast_like(s,w)
 p=div(w,wide_s)
 l=add(log(s),m)
 output w output s output p output l
}'''


def test_multioutput_normalization_is_generic_and_retains_three_sweep_semantics():
    p=compile_source(GENERIC).pipelines[0]
    assert len(p.triton.kernels)==1 and len(p.triton.kernels[0].results)==4
    src=p.triton.kernels[0].source
    assert src.count('    for ')==3 # max, sum/store w, then normalized full output
    actual=run(GENERIC,[[0,1,2,-2,-1,0]],{'B':2,'R':3})
    w=[math.exp(-2),math.exp(-1),1]
    assert actual['w']==pytest.approx(w*2,rel=1e-6)
    assert actual['s']==pytest.approx([sum(w)]*2,rel=1e-6)
    assert actual['p']==pytest.approx([v/sum(w) for v in w]*2,rel=1e-6)
    assert actual['l']==pytest.approx([math.log(sum(w))+2,math.log(sum(w))],rel=1e-6)


def test_independent_reductions_share_program_without_discarding_any_output():
    src='pipeline p { input x : fp32[2,3] m=max_last(x) s=sum_last(x) y=add(m,s) output m output s output y }'
    p=compile_source(src).pipelines[0]
    assert len(p.triton.kernels)==1
    assert run(src,[[1,2,3,4,5,6]])=={'m':[3,6],'s':[6,15],'y':[9,21]}


def test_equal_dimensions_do_not_turn_right_aligned_broadcast_into_row_broadcast():
    src='pipeline p { input x : fp32[2,2,2] input bias : fp32[2,2] b=broadcast_like(bias,x) m=max_last(x) s=sum_last(add(x,b)) y=add(m,s) output y }'
    got=run(src,[[0]*8,[1,2,3,4]])['y']
    assert got==[3,7,3,7]
    p=compile_source(src).pipelines[0]
    assert not any(g.reduction_domain and any(n.op=='broadcast_like' for n in g.members) for g in p.fusion.groups)


def test_shared_domain_traffic_counts_two_lse_sweeps_and_no_repeated_max_per_column():
    from flowgpu.compiler.traffic import domain_read_count,at_unit
    src=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()
    p=compile_source(src,shape_bindings={'R':7,'B':2,'H':3,'D':33},
                     cross_domain_strategy='materialize').pipelines[0]
    assert at_unit(domain_read_count(p.dataflow,p.fusion.groups[0],'lse_in'))==2


def test_external_late_gemm_is_not_read_before_it_executes():
    src='pipeline p { input x : fp32[2,3] input a : fp32[2,2] input b : fp32[2,3] e=exp(x) z=matmul(a,b) s=sum_last(z) w=broadcast_like(s,x) y=add(e,w) output y }'
    p=compile_source(src).pipelines[0]
    assert 'e' not in p.fusion.fused_values
    assert run(src,[[0]*6,[1]*4,[1]*6])['y']==[7]*6


def test_cross_row_indirect_access_stays_outside_a_reduction_domain():
    src='pipeline p { input x : fp32[2,3] input i : int32[2] m=max_last(x) y=gather_row_offset(m,i) output y }'
    p=compile_source(src).pipelines[0]
    assert not any(g.reduction_domain and any(n.op=='gather_row_offset' for n in g.members) for g in p.fusion.groups)
    assert run(src,[[1,2,3,4,5,6],[1,-1]])['y']==[6,3]


def test_literal_one_column_lse_keeps_finite_and_invalid_semantics():
    src=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()
    options={'shape_bindings':{'R':1,'B':1,'H':2,'D':3}}
    got=run(src,[[1,2,3,4,5,6],[5,-math.inf]],**options)
    assert got['out'][:3]==[1,2,3] and all(math.isnan(v) for v in got['out'][3:])
    assert got['out_lse']==[5,-math.inf]


def test_cli_cache_control_and_other_platforms(capsys,tmp_path):
    src=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()
    assert cached_module(src,cache_dir=tmp_path,reduction_fusion=False)[2]!=cached_module(src,cache_dir=tmp_path,reduction_fusion=True)[2]
    assert main(['examples/batch8/dcp_lse_combine.fgpu','--no-reduction-fusion','--emit','triton'])==0
    assert capsys.readouterr().out.count('@triton.jit')==6
    assert main(['examples/batch8/dcp_lse_combine.fgpu','--emit','json',
                 '--cross-domain-strategy','materialize'])==0
    data=json.loads(capsys.readouterr().out)['pipelines'][0]
    assert data['fusion']['kernels'][0]['reduction_domain']==['B','H','R']
    other=compile_source(src,platform_config=resolve_platform('kunlunxin')).pipelines[0]
    assert not other.triton.complete and not any(g.reduction_domain for g in other.fusion.groups)
