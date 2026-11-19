"""Synthetic observations test selection contracts, never performance claims."""
import copy
import json
from pathlib import Path
import pytest
from compiler.compile import compile_source
from compiler import reference
from compiler.cache import cached_module
from compiler.gather_strategy import compare_gather_strategies
from compiler.gather_tuning import select_paired, policy_identity, save_policy, load_policy
from compiler.cli import main

SOURCE='''pipeline p { input i : int64[N] input a : fp32[T,A] input b : int32[T,B]
 input c : int64[T] input d : fp16[T,D]
 oa=gather_rows(a,i) ob=gather_rows(b,i) oc=gather_rows(c,i) od=gather_rows(d,i)
 output oa output ob output oc output od }'''
BINDINGS=dict(N=3,T=2,A=2,B=3,D=7)
PLAN=[dict(outputs=['oa','ob','oc'],mapping='row_tiled',rows=8,columns=64),
      dict(outputs=['od'],mapping='flat',rows=None,columns=None)]


def test_partial_partition_preserves_values_lifetimes_and_schedule():
    p=compile_source(SOURCE,gather_plan=PLAN).pipelines[0]
    assert len(p.triton.kernels)==2 and len(p.memory.slots)==4
    ns={};exec(reference.module([p.reference]),ns)
    # Reference flat-list convention takes declared inputs followed by symbols.
    import inspect
    run=ns['run_p'];params=inspect.signature(run).parameters
    values=dict(i=[-1,0,2**40],a=[1,2,3,4],b=list(range(6)),c=[2**40,2**41],d=list(range(14)),**BINDINGS)
    got=run(**{k:values[k] for k in params})
    assert got['oa']==[3,4,1,2,0,0] and got['oc']==[2**41,2**40,0]
    assert got['od']==list(range(7,14))+list(range(7))+[0]*7
    r=compare_gather_strategies(SOURCE,BINDINGS,sm_count=108,partial=True)
    partial=next(x for x in r['candidates'] if x['name']=='partial-tiled8x64-flat')
    assert len(r['candidates'])==9
    assert [k['gather_mapping'] for k in partial['kernels']]==['row_tiled','flat']
    assert partial['decision_features']['max_spill'] is None


@pytest.mark.parametrize('bad',[
 [dict(outputs=['oa','oa'],mapping='flat',rows=None,columns=None)],
 [dict(outputs=['unknown'],mapping='flat',rows=None,columns=None)],
 [dict(outputs=['oa'],mapping='row_tiled',rows=128,columns=128)],
 [dict(outputs=['oa'],mapping='flat',rows=8,columns=64)],
 [dict(outputs=['oa'],mapping='row_tiled',rows=None,columns=64)],
])
def test_invalid_plans_refused(bad):
    with pytest.raises(ValueError, match='(?:gather plan outputs must name distinct bare gather SSA values|indexed R/D tile exceeds reduction_block_limit|flat gather plan cannot contain row/column tiles|row_tiled gather plan requires both tile dimensions)'):compile_source(SOURCE,gather_plan=bad)


def test_late_dependency_and_different_index_refused():
    src='pipeline p { input x : fp32[2,3] input i : int64[1] a=gather_rows(x,i) z=exp(x) b=gather_rows(z,i) output a output b }'
    plan=[dict(outputs=['a','b'],mapping='flat',rows=None,columns=None)]
    with pytest.raises(ValueError,match='late'):compile_source(src,gather_plan=plan)
    src='pipeline p { input x : fp32[2,3] input i : int64[1] input j : int64[1] a=gather_rows(x,i) b=gather_rows(x,j) output a output b }'
    with pytest.raises(ValueError,match='shared index'):compile_source(src,gather_plan=plan)
    with pytest.raises(ValueError,match='mutually'):compile_source(SOURCE,gather_plan=PLAN,gather_fusion=True)


def test_schedule_survives_indexed_consumer_and_cli_cache(tmp_path,capsys):
    src='pipeline p { input x : fp32[2,3] input i : int64[1] a=gather_rows(x,i) b=exp(a) output b }'
    p=compile_source(src,gather_plan=[dict(outputs=['a'],mapping='row_tiled',rows=4,columns=64)]).pipelines[0]
    assert len(p.triton.kernels)==2 and p.fusion.groups[0].row_gather_schedule==('row_tiled',4,64)
    a=cached_module(SOURCE,gather_plan=PLAN)[2]
    b=cached_module(SOURCE,gather_fusion=True)[2];assert a!=b
    source=tmp_path/'p.fgpu';source.write_text(SOURCE)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(PLAN))
    for cached in [False,True]:
        args=[str(source),'--emit','triton','--gather-plan',str(plan)]
        if cached:args+=['--cache-dir',str(tmp_path/'cache')]
        assert main(args)==0
        assert 'tl.program_id(1)' in capsys.readouterr().out


def observed():
    r=compare_gather_strategies(SOURCE,BINDINGS,sm_count=108,partial=True)
    device=dict(name='unit-test',uuid='fake-test-device',sm_count=108)
    context={k:dict(**v,stride=[1]*len(v['shape'])) for k,v in r['candidates'][0]['inputs'].items()}
    e=dict(schema_version=1,source_sha256=r['source_sha256'],compiler_fingerprint=r['compiler_fingerprint'],device=device,records=[])
    for i in r['candidates']:
        values=[10.,10.,10.,10.] if i['name']=='flat-separate' else [8.,8.,8.,8.]
        e['records'].append(dict(candidate=i['name'],bindings=BINDINGS,module_sha256=i['module_sha256'],
            input_context=context,kernels=[dict(name=k['name'],n_regs=32,n_spills_reported=0,shared_reported=0,num_warps=4) for k in i['kernels']],
            samples=dict(graph_us=values,wall_us=values),correctness_passed=True))
    return compare_gather_strategies(SOURCE,BINDINGS,sm_count=108,partial=True,evidence=e),e,context,device


def test_paired_gate_rejects_unstable_gain_and_missing_rounds():
    r,_,_,_=observed()
    assert select_paired(r,'graph_us')['selected']!='flat-separate'
    for i in r['candidates'][1:]:i['observation']['samples']['graph_us']=[7.,7.,7.,15.]
    assert select_paired(r,'graph_us')['selected']=='flat-separate'
    r['candidates'][0]['observation']['samples']['graph_us']=[10.]
    with pytest.raises(ValueError,match='four'):select_paired(r,'graph_us')


def test_policy_cache_rechecks_source_context_module_and_corruption(tmp_path):
    r,e,context,device=observed()
    identity=policy_identity(SOURCE,BINDINGS,device,context,'graph_us','unit-test',.05,.05)
    selected,path=save_policy(tmp_path,identity,r,e)
    assert load_policy(tmp_path,identity,SOURCE)==(selected,'hit')
    changed=copy.deepcopy(identity);changed['objective']='wall_us'
    assert load_policy(tmp_path,changed,SOURCE)[1]=='miss'
    changed=copy.deepcopy(identity);changed['input_context']['a']['stride']=[2,1]
    assert load_policy(tmp_path,changed,SOURCE)[1]=='miss'
    assert load_policy(tmp_path,identity,SOURCE.replace('fp32','bf16'))[0] is None
    path.write_text('{}');assert load_policy(tmp_path,identity,SOURCE)[0] is None


def test_cache_does_not_trust_changed_selection_even_with_valid_digest(tmp_path):
    from compiler.gather_tuning import digest
    r,e,context,device=observed()
    identity=policy_identity(SOURCE,BINDINGS,device,context,'wall_us','unit-test',.05,.05)
    _,path=save_policy(tmp_path,identity,r,e);entry=json.loads(path.read_text())
    entry['payload']['selection']['selected']='flat-separate';entry['sha256']=digest(entry['payload'])
    path.write_text(json.dumps(entry));assert load_policy(tmp_path,identity,SOURCE)[1]=='selection mismatch'
