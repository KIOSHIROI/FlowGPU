"""Execute emitted two-stage programs; device validation remains separate."""
import ast
from dataclasses import replace
import math
from pathlib import Path

import pytest

from compiler.compile import compile_source
from compiler.backend.platform_configs import resolve_platform
from compiler.backend.triton_backend import module
from compiler.optimization.strategy import candidates,compare_strategies
from compiler.optimization.strategy_tuning import policy_identity
from compiler.optimization.measurement import digest
from compiler.runtime.cache import cached_module
from test.segmented_support import Buffer,inputs,oracle,check,execute

SOURCE_PATH=Path(__file__).resolve().parent/'source.fgpu'
if not SOURCE_PATH.exists():SOURCE_PATH=Path(__file__).resolve().parents[1]/'examples/batch8/dcp_lse_combine.fgpu'
SOURCE=SOURCE_PATH.read_text()


def merged(source=SOURCE):
    return compile_source(source,cross_domain_strategy='segmented-merge',prune_kernel_abi=True,
        platform_config=resolve_platform(indexed_columns=4,indexed_rows=2)).pipelines[0]


@pytest.mark.parametrize('shape',[(1,1,2,1),(7,2,3,5),(17,1,2,9),(33,2,1,7)])
@pytest.mark.parametrize('distribution',['normal','peaky','mixed-invalid'])
@pytest.mark.parametrize('base2',[False,True])
def test_emitted_merged_program(shape,distribution,base2):
    source=SOURCE.replace('exp(shifted)','exp2(shifted)').replace('log(denom)','log2(denom)') if base2 else SOURCE
    source=source.replace('pipeline lse','pipeline different').replace('weights','importance')
    program=merged(source);args,values,scores=inputs(shape,distribution);before=[list(a.data) for a in args]
    got,launches=execute(module([program.triton]),args,pipeline='different')
    check(got,oracle(shape,values,scores,base2));assert len(launches)==2
    for arg,old in zip(args,before):assert all(x==y or x!=x and y!=y for x,y in zip(arg.data,old))


@pytest.mark.parametrize('value',[math.nan,math.inf,-math.inf])
@pytest.mark.parametrize('score',[0.,-80.,-110.])
def test_nonfinite_payload_and_fp32_underflow_replay(value,score):
    program=merged();shape=(9,1,1,1)
    payload=[0.]*9;payload[2]=value;scores=[-80.]*4+[0.]*5;scores[2]=score
    got,_=execute(module([program.triton]),(Buffer(payload,shape,'fp16'),Buffer(scores,(9,1,1),'fp32')))
    # FP32 exp(-110) is zero: the source expression must preserve 0*Inf=NaN.
    if math.isnan(value) or score==-110.:assert math.isnan(got['out'][0])
    else:assert got['out'][0]==value


def test_replay_is_necessary_and_local_fp32_overflow_is_guarded():
    program=merged();code=module([program.triton]);tree=ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='needs_replay' for t in n.targets):
            n.value=ast.parse('tl.full((), False, tl.int1)',mode='eval').body
    broken=ast.unparse(ast.fix_missing_locations(tree));args=(Buffer([0.,0.,math.inf]+[0.]*6,(9,1,1,1),'fp16'),Buffer([-80.,-80.,-110.,-80.]+[0.]*5,(9,1,1),'fp32'))
    good,_=execute(code,args);bad,_=execute(broken,args)
    assert math.isnan(good['out'][0]) and bad['out'][0]==math.inf
    source=SOURCE.replace('recv : fp16','recv : fp32');program=merged(source)
    payload=[1e38]*4+[0.]*13;scores=[-80.]*4+[-1000.]*12+[0.]
    got,_=execute(module([program.triton]),(Buffer(payload,(17,1,1,1),'fp32'),Buffer(scores,(17,1,1),'fp32')))
    expected=4e38*math.exp(-80.)/(1+4*math.exp(-80.))
    assert got['out'][0]==pytest.approx(expected,rel=.003)


def test_proof_stage_storage_traffic_and_cache_identity(tmp_path):
    program=merged();plan=program.memory.stage_plan;plan.validate(program.dataflow)
    assert [s.kind for s in plan.stages]==['local','merge']
    assert program.kernel_schedule.dependencies==((),(0,))
    assert all(s.writer==0 and s.readers==(1,) for s in plan.scratch) and len(plan.scratch)==4
    with pytest.raises(ValueError, match='invalid shifted-exponential merge proof'):replace(plan,merge=replace(plan.merge,exp_op='log')).validate(program.dataflow)
    b=dict(R=1025,B=1,H=2,D=7)
    report=compare_strategies(SOURCE,b,recompute=True,segmented=True,merge_statistics=True,sm_count=114)
    item=report['candidates'][-1];assert item['name']=='merged-r256-d8' and item['launches']==2
    assert item['normal_path_logical_traffic_bytes']+item['conditional_replay_logical_traffic_bytes']==item['logical_traffic_bytes']
    assert item['conditional_replay_logical_traffic_bytes']>0
    assert item['kernels'][1]['exceptional_replay']
    assert item['kernels'][1]['original_wide_full_r_scans']==0 and item['kernels'][1]['conditional_original_wide_full_r_scans']==1
    device=dict(index=0);identity=policy_identity(SOURCE,'lse',b,{},device,objective='graph_us',workload_tag='CPU-only',recompute=True,segmented=True,merge_statistics=True)
    without=policy_identity(SOURCE,'lse',b,{},device,objective='graph_us',workload_tag='CPU-only',recompute=True,segmented=True)
    assert digest(identity)!=digest(without)
    a=cached_module(SOURCE,cache_dir=tmp_path,cross_domain_strategy='segmented')
    b=cached_module(SOURCE,cache_dir=tmp_path,cross_domain_strategy='segmented-merge')
    assert a[2]!=b[2] and a[0]!=b[0]
    with pytest.raises(ValueError, match='merged statistics require explicit segmented opt-in'):candidates(merge_statistics=True)


def test_proof_refuses_nonlinear_or_different_weight_expressions():
    changes=[SOURCE.replace('weights = exp(shifted)','weights = add(exp(shifted),ones_like(shifted))'),
             SOURCE.replace('weighted = mul(values, wide_w)','weighted = mul(wide_w,wide_w)'),
             SOURCE.replace('weighted = mul(values, wide_w)','weighted = add(values,wide_w)'),
             SOURCE.replace('m = where(mvalid, m0, zero)','m = log(m0)'),
             SOURCE.replace('weighted = mul(values, wide_w)','weighted = mul(add(values,broadcast_like(reshape(m0,[B,H,1,1]),values)),wide_w)')]
    for source in changes:
        with pytest.raises(ValueError, match='two-stage merge'):merged(source)


def test_unprotected_row_major_max_and_sum_outputs():
    source='''pipeline generic {
      input score: fp32[G,R] input payload: fp32[G,D,R]
      maximum=max_last(score) weights=exp(sub(score,broadcast_like(maximum,score)))
      denominator=sum_last(weights)
      expanded=broadcast_like(reshape(weights,[G,1,R]),payload)
      numerator=sum_last(mul(payload,expanded))
      result=div(numerator,broadcast_like(reshape(denominator,[G,1,1]),numerator))
      output result output denominator output maximum
    }'''
    p=merged(source);score=[float(i%7-3) for i in range(34)];payload=[float(i%11-5) for i in range(102)]
    got,_=execute(module([p.triton]),(Buffer(score,(2,17),'fp32'),Buffer(payload,(2,3,17),'fp32')),pipeline='generic')
    for gi in range(2):
        maximum=max(score[gi*17:(gi+1)*17]);w=[math.exp(x-maximum) for x in score[gi*17:(gi+1)*17]]
        assert got['maximum'][gi]==maximum and got['denominator'][gi]==pytest.approx(sum(w),rel=.003)
        for di in range(3):assert got['result'][gi*3+di]==pytest.approx(sum(payload[(gi*3+di)*17+ri]*w[ri] for ri in range(17))/sum(w),abs=.003,rel=.003)
    got,_=execute(module([p.triton]),(Buffer([-math.inf]*34,(2,17),'fp32'),Buffer(payload,(2,3,17),'fp32')),pipeline='generic')
    assert all(math.isnan(v) for v in got['result']+got['denominator'])
