import ast
import json
import math

import pytest

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.platform_configs import resolve_platform
from flowgpu.compiler import reference
from flowgpu.compiler.triton_backend import module
from flowgpu.compiler.traffic import domain_read_count, at_unit
from flowgpu.compiler.cache import cached_module
from flowgpu.compiler.cli import main


POOL = '''pipeline pool {
 input kv : fp32[B,D,K] input scores : fp32[B,D,K]
 v=to_fp32(kv) s=to_fp32(scores) m=max_last(s)
 mb=broadcast_like(m,s) e=exp(sub(s,mb))
 z=sum_last(e) n=sum_last(mul(v,e))
 out=cast_like(div(n,z),kv) output out
}'''


def config(**kwargs):
    return resolve_platform('cuda', reduction_group_limit=3,
                            coalesce_reduction_sweeps=True, **kwargs)


def test_streaming_region_keeps_dependency_schedule_and_cost_model_in_agreement():
    p=compile_source(POOL,platform_config=config()).pipelines[0]
    assert len(p.triton.kernels)==1 and len(p.memory.slots)==1
    group=p.fusion.groups[0]
    assert group.reduction_batches == (('m',), ('z','n'))
    assert p.triton.kernels[0].source.count('    for ')==2
    text=p.triton.kernels[0].source
    assert 'row_coordinate' in text
    assert 'scores_address = (row_coordinate[None, :]) * scores_stride0' in text
    assert '// (scores_dim2)' not in text
    assert at_unit(domain_read_count(p.dataflow,group,'scores'))==2
    assert at_unit(domain_read_count(p.dataflow,group,'kv'))==1
    ast.parse(module([p.triton]))


@pytest.mark.parametrize('k',[1,3,7,33])
def test_streaming_reference_matches_independent_stable_pool(k):
    p=compile_source(POOL,platform_config=config()).pipelines[0]
    scope={};exec(reference.module([p.reference]),scope)
    scores=[(i%5)-2 for i in range(4*k)]
    values=[(i%7)-3 for i in range(4*k)]
    got=scope['run_pool'](values,scores,2,2,k)['out']
    want=[]
    for row in range(4):
        ss=scores[row*k:(row+1)*k];vv=values[row*k:(row+1)*k]
        m=max(ss);w=[math.exp(v-m) for v in ss]
        want.append(sum(a*b for a,b in zip(vv,w))/sum(w))
    assert got==pytest.approx(want,rel=1e-6,abs=1e-6)


def test_computed_row_dependency_prevents_unsafe_scan_coalescing():
    src='''pipeline p { input x:fp32[B,K] m=max_last(x)
        e=exp(sub(x,broadcast_like(m,x))) s=sum_last(e)
        normalized=div(e,broadcast_like(s,e)) q=sum_last(normalized)
        output q }'''
    p=compile_source(src,platform_config=config()).pipelines[0]
    group=p.fusion.groups[0]
    assert group.reduction_batches == (('m',),('s',),('q',))
    assert p.triton.kernels[0].source.count('    for ')==3
    scope={};exec(reference.module([p.reference]),scope)
    assert scope['run_p']([0,1,2,-2,-1,0],2,3)['q']==pytest.approx([1,1])


def test_opt_in_schedule_and_foreign_region_options_are_part_of_cache_key(tmp_path):
    _,_,old=cached_module(POOL,cache_dir=tmp_path)
    _,_,new=cached_module(POOL,cache_dir=tmp_path,platform_config=config())
    assert old != new
    assert len(compile_source(POOL).pipelines[0].triton.kernels)>1
    foreign=resolve_platform('kunlunxin',allowed_kernel_kinds=('primitive','pointwise','layout'),
                             reduction_group_limit=3,coalesce_reduction_sweeps=True,
                             experimental_region_fusion=True,indexed_columns=32,indexed_rows=4)
    assert len(compile_source(POOL,platform_config=foreign).pipelines[0].triton.kernels)==1


def test_cli_can_request_the_explicit_schedule(tmp_path,capsys):
    path=tmp_path/'pool.fgpu';path.write_text(POOL)
    assert main([str(path),'--reduction-group-limit','3','--coalesce-reduction-sweeps','--emit','triton'])==0
    assert capsys.readouterr().out.count('@triton.jit')==1
    assert main([str(path),'--reduction-group-limit','3','--coalesce-reduction-sweeps','--emit','json'])==0
    data=json.loads(capsys.readouterr().out)
    assert data['pipelines'][0]['fusion']['kernels'][0]['reduction_batches']==[['m'],['z','n']]


@pytest.mark.parametrize('value',[0,-1,True])
def test_reduction_limit_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='reduction_group_limit must be a positive integer'):resolve_platform('cuda',reduction_group_limit=value)
