"""Exercise emitted indices/stores and public policy contracts on CPU.

The standard-library device API double is not Triton compilation or GPU proof.
"""
import ast
from dataclasses import asdict, replace
import math
from pathlib import Path

import pytest

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.backend.platform_configs import resolve_platform
from flowgpu.compiler.backend.triton_backend import module,device_key_source
from flowgpu.compiler.backend.specialization import freeze_autotune
from flowgpu.compiler.analysis.traffic import plan_traffic,ratio_at
from flowgpu.compiler.optimization.strategy import candidates,compare_strategies,launch_value
from flowgpu.compiler.optimization.strategy_tuning import policy_identity,bound_comparison,load_policy,_prepared_proof
from flowgpu.compiler.optimization.measurement import digest,select_paired,_atomic_json
from flowgpu.compiler.runtime.cache import cached_module
from flowgpu.compiler.frontend.diagnostic import CompileError
from test.segmented_support import inputs,oracle,check,execute,Buffer

SOURCE=Path(__file__).resolve().parents[1].joinpath('examples/batch8/dcp_lse_combine.fgpu').read_text()


def compile_staged(source=SOURCE,rows=4,columns=2):
    return compile_source(source,cross_domain_strategy='segmented',prune_kernel_abi=True,
        platform_config=resolve_platform(indexed_columns=rows,indexed_rows=columns)).pipelines[0]


@pytest.mark.parametrize('shape',[(1,1,2,1),(7,2,3,5),(17,1,2,9),(33,2,1,7)])
@pytest.mark.parametrize('distribution',['normal','peaky','mixed-invalid'])
@pytest.mark.parametrize('base2',[False,True])
def test_emitted_stages_against_independent_scalar_oracle(shape,distribution,base2):
    text=SOURCE.replace('exp(shifted)','exp2(shifted)').replace('log(denom)','log2(denom)') if base2 else SOURCE
    text=text.replace('pipeline lse','pipeline unrelated').replace('weights','importance')
    p=compile_staged(text);args,values,scores=inputs(shape,distribution)
    before=[list(a.data) for a in args]
    got,launches=execute(module([p.triton]),args,pipeline='unrelated')
    check(got,oracle(shape,values,scores,base2))
    bindings=dict(zip(('R','B','H','D'),shape))
    assert [tuple(g) for _,g in launches]==[launch_value(s.grid,bindings) for s in p.memory.stage_plan.stages]
    for a,old in zip(args,before):
        assert all(x==y or x!=x and y!=y for x,y in zip(a.data,old))


def test_stage_ir_storage_traffic_and_prepared_host_share_one_plan():
    p=compile_staged(rows=256,columns=8);stages=p.memory.stage_plan
    stages.validate(p.dataflow)
    assert [s.kind for s in stages.stages]==['max','partials','finish']
    assert p.kernel_schedule.dependencies==((),(0,),(1,))
    b=dict(R=1025,B=2,H=3,D=17)
    report=compare_strategies(SOURCE,b,recompute=True,segmented=True,sm_count=114,compile_options={'prune_kernel_abi':True})
    item=report['candidates'][-1]
    assert item['segmented_applied'] and item['intermediate_bytes']==4*6*5*(17+2)
    assert item['allocated_bytes']==item['allocated_slot_bytes']+item['intermediate_bytes']
    assert [k['grid'] for k in item['kernels']]==[[6,5],[6,3,5],[6,3]]
    traffic=plan_traffic(p.dataflow,p.fusion,p.memory,tiled=True)
    assert ratio_at(traffic,b)>0
    assert {s.name for s in stages.scratch}<={c.value for c in traffic.charges}
    context=[dict(shape=[17,2,3,5],stride=[42,21,7,1],dtype='torch.float16',device='cuda:0',alignment_mod128=2),
             dict(shape=[17,2,3],stride=[10,5,1],dtype='torch.float32',device='cuda:0',alignment_mod128=4)]
    code=module([p.triton])+device_key_source();fixed=freeze_autotune(code,{},pipeline='lse',context=context,device_key='CPU-only')
    assert 'prepared input context changed' in fixed
    def bodies(text):return [ast.dump(ast.Module(body=n.body,type_ignores=[])) for n in ast.parse(text).body
        if isinstance(n,ast.FunctionDef) and n.decorator_list]
    assert bodies(code)==bodies(fixed)
    assert asdict(stages)['scratch'][0]['writer']==0


def test_unsupported_regions_and_stage_corruption_fail_closed():
    dependent=SOURCE.replace('weighted = mul(values, wide_w)',
        'early=div(wide_w,broadcast_like(reshape(denom,[B,H,1,1]),wide_w)) weighted=mul(values,early)')
    for text in (dependent,SOURCE.replace('output out\n','output weights\noutput out\n'),
                 'pipeline p { input x: fp32[7] y=exp(x) output y }',
                 SOURCE.replace('denom = sum_last(weights)','low=cast_like(weights,recv) denom=sum_last(low)')):
        with pytest.raises((ValueError,CompileError)):compile_staged(text)
    p=compile_staged();plan=p.memory.stage_plan
    damaged=replace(plan,stages=(plan.stages[1],plan.stages[0],plan.stages[2]))
    with pytest.raises(ValueError, match='invalid segmented stage order'):damaged.validate(p.dataflow)
    damaged=replace(plan,scratch=(replace(plan.scratch[0],writer=99),)+plan.scratch[1:])
    with pytest.raises(ValueError, match='scratch lifetime disagrees with stage dataflow'):damaged.validate(p.dataflow)
    args,_,_=inputs((1025,1,1,1),'normal')
    with pytest.raises(ValueError,match='partial rows'):execute(module([p.triton]),args)


def test_opt_in_cache_and_empirical_selection_include_segmented(tmp_path):
    assert not any(s.mode=='segmented' for s in candidates(recompute=True,large_tiles=True))
    with pytest.raises(ValueError, match='segmented candidates require explicit recompute opt-in'):candidates(segmented=True)
    default=cached_module(SOURCE,cache_dir=tmp_path/'sources')
    staged=cached_module(SOURCE,cache_dir=tmp_path/'sources',cross_domain_strategy='segmented')
    assert default[2]!=staged[2] and default[0]!=staged[0]
    assert cached_module(SOURCE,cache_dir=tmp_path/'sources',cross_domain_strategy='segmented')[1]
    b=dict(R=1025,B=1,H=2,D=7)
    report=compare_strategies(SOURCE,b,recompute=True,segmented=True,sm_count=114)
    device=dict(name='CPU-double',uuid='not-a-device',sm_count=114,total_memory=80,capability=[9,0],
                index=0,torch='synthetic',cuda='synthetic',triton='synthetic')
    context={n:dict(**v,stride=[1]*len(v['shape']),alignment_mod128=0)
             for n,v in report['candidates'][0]['inputs'].items()}
    identity=policy_identity(SOURCE,'lse',b,context,device,objective='graph_us',workload_tag='cpu-contract',
        recompute=True,segmented=True,rounds=4)
    without=policy_identity(SOURCE,'lse',b,context,device,objective='graph_us',workload_tag='cpu-contract',recompute=True,rounds=4)
    assert digest(identity)!=digest(without)
    evidence=dict(schema_version=1,source_sha256=identity['source_sha256'],compiler_fingerprint=identity['compiler_fingerprint'],device=device,records=[])
    for spec,item in zip(candidates(recompute=True,segmented=True),report['candidates']):
        p=compile_source(SOURCE,**spec.options()).pipelines[0]
        proof=_prepared_proof(module([p.triton]),'lse',context,device,[n.output for n in p.dataflow.nodes if n.op=='input'])
        evidence['records'].append(dict(candidate=spec.name,bindings=b,module_sha256=item['module_sha256'],deployment=proof,
            input_context=context,kernels=[dict(name=k['name'],n_regs=32,n_spills_reported=0,shared_reported=256,num_warps=4) for k in item['kernels']],
            samples=dict(graph_us=[5. if spec.mode=='segmented' else 10.]*4,wall_us=[20.]*4),correctness_passed=True))
    # Synthetic timing validates dispatch/cache contracts only, never speed.
    comparison=bound_comparison(SOURCE,identity,evidence)
    selected=select_paired(comparison,'graph_us')
    assert selected['selected']=='segmented-r256-d8'
    path=tmp_path/'policy.json';payload=dict(identity=identity,selection=selected,evidence=evidence)
    _atomic_json(path,dict(payload=payload,sha256=digest(payload)))
    assert load_policy(path,SOURCE,identity)[1]=='hit'
    assert load_policy(path,SOURCE,without)[0] is None


def test_row_major_program_and_index_mutation():
    text='''pipeline average {
      input score: fp32[G,R] input payload: fp32[G,D,R]
      maximum=max_last(score) weights=exp(sub(score,broadcast_like(maximum,score)))
      denominator=sum_last(weights)
      expanded=broadcast_like(reshape(weights,[G,1,R]),payload)
      numerator=sum_last(mul(payload,expanded))
      result=div(numerator,broadcast_like(reshape(denominator,[G,1,1]),numerator))
      output result
    }'''
    p=compile_staged(text)
    score=[float(i%7-3) for i in range(34)];payload=[float(i%11-5) for i in range(102)]
    got,_=execute(module([p.triton]),(Buffer(score,(2,17),'fp32'),Buffer(payload,(2,3,17),'fp32')),pipeline='average')
    expected=[]
    for gi in range(2):
        scores=score[gi*17:(gi+1)*17];mx=max(scores);weights=[math.exp(x-mx) for x in scores];den=sum(weights)
        expected.extend(sum(payload[(gi*3+di)*17+ri]*weights[ri] for ri in range(17))/den for di in range(3))
    assert got['result']==pytest.approx(expected,abs=.003,rel=.003)
    p=compile_staged();code=module([p.triton])
    broken=code.replace('segment * FG_R_TILE + step * FG_R_TILE + r','step * FG_R_TILE + r')
    assert broken!=code
    args,values,scores=inputs((17,1,2,5),'normal')
    with pytest.raises(AssertionError, match="\\('out', -0\\.3916015625, -0\\.67626953125\\)"):
        got,_=execute(broken,args);check(got,oracle((17,1,2,5),values,scores))
