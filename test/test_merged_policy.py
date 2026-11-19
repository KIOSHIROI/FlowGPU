"""Synthetic cache evidence tests are contracts, never performance evidence."""
from pathlib import Path

from compiler.compile import compile_source
from compiler.backend.triton_backend import module
from compiler.optimization.strategy import candidates,compare_strategies
from compiler.optimization.strategy_tuning import policy_identity,bound_comparison,load_policy,_prepared_proof
from compiler.optimization.measurement import digest,select_paired,_atomic_json


def test_merged_policy_roundtrip_and_stale_opt_in_rejection(tmp_path):
    source_path=Path(__file__).resolve().parent/'source.fgpu'
    if not source_path.exists():source_path=Path(__file__).resolve().parents[1]/'examples/batch8/dcp_lse_combine.fgpu'
    source=source_path.read_text()
    bindings=dict(R=1025,B=1,H=2,D=7)
    report=compare_strategies(source,bindings,recompute=True,large_tiles=True,segmented=True,merge_statistics=True,sm_count=114)
    device=dict(name='CPU-double',uuid='not-a-device',sm_count=114,total_memory=80,capability=[9,0],index=0,torch='synthetic',cuda='synthetic',triton='synthetic')
    context={name:dict(**v,stride=[1]*len(v['shape']),alignment_mod128=0) for name,v in report['candidates'][0]['inputs'].items()}
    options=dict(objective='graph_us',workload_tag='CPU-contract',recompute=True,large_tiles=True,segmented=True,rounds=4)
    identity=policy_identity(source,'lse',bindings,context,device,merge_statistics=True,**options)
    without=policy_identity(source,'lse',bindings,context,device,**options)
    evidence=dict(schema_version=1,source_sha256=identity['source_sha256'],compiler_fingerprint=identity['compiler_fingerprint'],device=device,records=[])
    for spec,item in zip(candidates(recompute=True,large_tiles=True,segmented=True,merge_statistics=True),report['candidates']):
        p=compile_source(source,**spec.options()).pipelines[0]
        proof=_prepared_proof(module([p.triton]),'lse',context,device,[n.output for n in p.dataflow.nodes if n.op=='input'])
        evidence['records'].append(dict(candidate=spec.name,bindings=bindings,module_sha256=item['module_sha256'],deployment=proof,input_context=context,
            kernels=[dict(name=k['name'],n_regs=32,n_spills_reported=0,shared_reported=256,num_warps=4) for k in item['kernels']],
            samples=dict(graph_us=[5. if spec.mode=='segmented-merge' else 10.]*4,wall_us=[20.]*4),correctness_passed=True))
    selection=select_paired(bound_comparison(source,identity,evidence),'graph_us')
    assert selection['selected']=='merged-r256-d8'
    path=tmp_path/'policy.json';payload=dict(identity=identity,selection=selection,evidence=evidence)
    _atomic_json(path,dict(payload=payload,sha256=digest(payload)))
    assert load_policy(path,source,identity)[1]=='hit' and load_policy(path,source,without)[0] is None
