"""Synthetic latency/resource observations verify contracts, not GPU speedups."""
import copy
import json
from pathlib import Path

import pytest

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.backend.triton_backend import module
from flowgpu.compiler.optimization.strategy import candidates,compare_strategies
from flowgpu.compiler.optimization.measurement import digest,select_paired,_atomic_json
from flowgpu.compiler.optimization.strategy_tuning import policy_identity,bound_comparison,load_policy,_prepared_proof

SOURCE=Path('examples/batch8/dcp_lse_combine.fgpu').read_text()


def observed(*,recompute=True,objective='graph_us'):
    bindings=dict(R=3073,B=1,H=2,D=3)
    report=compare_strategies(SOURCE,bindings,recompute=recompute,sm_count=108)
    device=dict(name='synthetic',uuid='unit-only',sm_count=108,total_memory=80,capability=[8,0],
                index=0,torch='synthetic',cuda='synthetic',triton='synthetic')
    inputs=report['candidates'][0]['inputs']
    context={n:dict(**v,stride=[1]*len(v['shape']),alignment_mod128=0) for n,v in inputs.items()}
    identity=policy_identity(SOURCE,'lse',bindings,context,device,objective=objective,workload_tag='unit',recompute=recompute,rounds=4)
    evidence=dict(schema_version=1,source_sha256=identity['source_sha256'],compiler_fingerprint=identity['compiler_fingerprint'],device=device,records=[])
    for spec,item in zip(candidates(recompute=recompute),report['candidates']):
        p=compile_source(SOURCE,**spec.options()).pipelines[0]
        proof=_prepared_proof(module([p.triton]),'lse',context,device,[n.output for n in p.dataflow.nodes if n.op=='input'])
        evidence['records'].append(dict(candidate=spec.name,bindings=bindings,module_sha256=item['module_sha256'],deployment=proof,
            input_context=context,kernels=[dict(name=k['name'],n_regs=32,n_spills_reported=0,shared_reported=256,num_warps=4) for k in item['kernels']],
            samples=dict(graph_us=[10. if spec.mode=='materialize' else 20.]*4,wall_us=[50. if spec.mode=='materialize' else 30.]*4),correctness_passed=True))
    return identity,evidence


def choose(identity,evidence):
    return select_paired(bound_comparison(SOURCE,identity,evidence),identity['objective'])


def test_graph_and_wall_can_select_opposite_policies_and_explain_scans():
    identity,evidence=observed()
    assert choose(identity,evidence)['selected']=='materialize'
    wall=copy.deepcopy(identity);wall['objective']='wall_us'
    selected=choose(wall,evidence)
    assert selected['selected'].startswith('recompute-')
    assert digest(identity)!=digest(wall)
    features=selected['decisions'][-1]['features']
    assert features['kernels'][0]['programs_per_sm'] and features['kernels'][0]['chunks_per_sweep']
    assert features['kernels'][0]['resources']['n_spills_reported']==0
    assert bound_comparison(SOURCE,identity,evidence)['execution_surface']=='prepared_codegen'


def test_opt_in_candidate_space_and_unstable_or_unknown_resources_keep_default():
    identity,evidence=observed(recompute=False,objective='wall_us')
    assert choose(identity,evidence)['selected']=='materialize'
    assert all(r['candidate'].startswith('materialize') for r in evidence['records'])
    identity,evidence=observed(objective='wall_us')
    for record in evidence['records'][2:]:record['samples']['wall_us']=[20,20,20,70]
    assert choose(identity,evidence)['selected']=='materialize'
    for record in evidence['records'][2:]:
        record['samples']['wall_us']=[20]*4;record['kernels'][0]['n_regs']=None
    assert choose(identity,evidence)['selected']=='materialize'


@pytest.mark.parametrize('field',['source','prepared_source','stride','alignment','device','resources','rounds','nan'])
def test_unbound_evidence_is_rejected(field):
    identity,evidence=observed();record=evidence['records'][-1]
    if field=='source':evidence['source_sha256']='stale'
    elif field=='prepared_source':record['deployment']['source_sha256']='stale'
    elif field=='stride':record['input_context']=copy.deepcopy(record['input_context']);record['input_context']['recv']['stride'][0]+=1
    elif field=='alignment':record['deployment']['context'][0]['alignment_mod128']=16
    elif field=='device':evidence['device']=dict(evidence['device'],uuid='other')
    elif field=='resources':record['kernels'][0]['n_regs']=-1
    elif field=='rounds':record['samples']['wall_us']=[1.]
    else:record['samples']['wall_us'][0]=float('nan')
    with pytest.raises(ValueError, match="(?:candidate/context ordering mismatch|evidence source/flowgpu/compiler/device mismatch|paired round count mismatch|prepared execution source mismatch|stale or incomplete observations: recompute-r1024-d4 \\['invalid or missing wall_us samples'\\]|stale or incomplete observations: recompute-r1024-d4 \\['invalid resource counters'\\])"):bound_comparison(SOURCE,identity,evidence)


def test_cache_reconstructs_hosts_and_reselects_instead_of_trusting_digest(tmp_path):
    identity,evidence=observed(objective='wall_us');selection=choose(identity,evidence)
    payload=dict(identity=identity,selection=selection,evidence=evidence)
    path=tmp_path/(digest(identity)+'.json');_atomic_json(path,dict(payload=payload,sha256=digest(payload)))
    assert load_policy(path,SOURCE,identity)==(payload,'hit')
    # JSON key order does not alter host ABI or generated source hashes.
    path.write_text(json.dumps(json.loads(path.read_text()),sort_keys=True))
    assert load_policy(path,SOURCE,identity)[1]=='hit'
    payload['selection']['selected']='materialize';_atomic_json(path,dict(payload=payload,sha256=digest(payload)))
    assert load_policy(path,SOURCE,identity)[1]=='selection mismatch'
    payload['selection']=selection;payload['evidence']['records'][0]['deployment']['source_sha256']='wrong'
    _atomic_json(path,dict(payload=payload,sha256=digest(payload)))
    assert load_policy(path,SOURCE,identity)[0] is None


@pytest.mark.parametrize('kw',[
    dict(objective='latency'),dict(large_tiles=True,recompute=False),dict(rounds=3),dict(graph_replays=0),
    dict(compile_options={'autotune':True}),dict(compile_options={'shape_bindings':{'R':3}}),
])
def test_invalid_policy_options_do_not_start_calibration(kw):
    options=dict(objective='graph_us',workload_tag='unit');options.update(kw)
    with pytest.raises(ValueError, match='(?:objective must be graph_us or wall_us|large tile comparison requires explicit recompute opt-in|at least four rounds and positive iteration counts required|strategy tuning requires fixed native sequential candidates|candidate configurations and symbolic bindings are owned by strategy tuning)'):policy_identity(SOURCE,'lse',{}, {}, {},**options)


@pytest.mark.parametrize('index',[1,2,3,4])
def test_nonselected_candidate_host_tampering_is_rejected(index):
    identity,evidence=observed()
    # Graph retains materialize. Every alternative must still be checked, even
    # when its latency cannot win and the outer payload digest is recomputed.
    evidence['records'][index]['deployment']['source_sha256']='altered-host'
    with pytest.raises(ValueError,match='prepared execution source mismatch'):
        bound_comparison(SOURCE,identity,evidence)


@pytest.mark.parametrize('field',['stride','alignment','device'])
def test_matching_new_identity_and_records_cannot_reuse_old_deployment(field):
    identity,evidence=observed()
    identity=copy.deepcopy(identity);evidence=copy.deepcopy(evidence)
    if field=='device':
        identity['device']['uuid']='another-device'
        evidence['device']=copy.deepcopy(identity['device'])
    else:
        context=copy.deepcopy(identity['input_context'])
        if field=='stride':context['recv']['stride'][0]+=1
        else:context['recv']['alignment_mod128']=16
        identity['input_context']=context
        for record in evidence['records']:record['input_context']=copy.deepcopy(context)
    with pytest.raises(ValueError,match='prepared execution source mismatch'):
        bound_comparison(SOURCE,identity,evidence)
