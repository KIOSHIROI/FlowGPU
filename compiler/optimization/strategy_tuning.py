"""Opt-in, measured reduction strategy selection on guarded prepared code.

All candidates receive the same deployment treatment. Graph and batch wall
objectives are separate cache identities. No unseen-shape prediction or R cutoff
is introduced. CUDA imports occur only in the runtime entry point.
"""
import ast
import hashlib
import json
from pathlib import Path
import time

from compiler.shared import compiler_fingerprint
from compiler.optimization.measurement import digest, select_paired, _atomic_json, probe_resources, measure_calls, GRAPH_RETENTION_BUDGET_BYTES
from compiler.optimization.strategy import candidates, compare_strategies, _compare_strategies
from compiler.backend.specialization import freeze_autotune
from compiler.backend.triton_backend import module, device_key_source
from compiler.optimization.preparation_cost import PhaseLedger


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _device_key(device):
    return repr((device['name'],device['total_memory'],device['sm_count'],
                 *device['capability'],device['uuid'],device['torch'],device['cuda'],device['triton']))


def _prepared_proof(code, pipeline, context, device, input_names):
    """Regenerate the measured host, retaining numerical kernel definitions."""
    original=_sha(code)
    if not any(isinstance(n,ast.FunctionDef) and n.name=='_flowgpu_device_key' for n in ast.parse(code).body):
        code+=device_key_source()
    inverse={'fp16':'torch.float16','bf16':'torch.bfloat16','fp32':'torch.float32',
             'int32':'torch.int32','int64':'torch.int64','bool':'torch.bool'}
    metadata=[dict(shape=v['shape'],stride=v['stride'],dtype=inverse[v['dtype']],
                   device='cuda:'+str(device['index']),alignment_mod128=v['alignment_mod128']) for v in (context[n] for n in input_names)]
    fixed=freeze_autotune(code,{},pipeline=pipeline,context=metadata,device_key=_device_key(device))
    return dict(original_module_sha256=original,module_sha256=_sha(code),source_sha256=_sha(fixed),
                context=metadata,device_key=_device_key(device),selections={})


def policy_identity(source, pipeline, bindings, context, device, *, objective, workload_tag,
                    recompute=False, large_tiles=False, segmented=False, merge_statistics=False, compile_options=None,
                    min_gain=.05, max_round_regression=.05, rounds=6, graph_replays=200, wall_calls=50,
                    reuse_calibration_graphs=False):
    if objective not in {'graph_us','wall_us'}:raise ValueError('objective must be graph_us or wall_us')
    if not isinstance(workload_tag,str) or not workload_tag.strip():raise ValueError('a workload tag is required')
    if not 0<=min_gain<1 or not 0<=max_round_regression<1:raise ValueError('invalid paired selection thresholds')
    if type(rounds) is not int or rounds<4 or any(type(n) is not int or n<=0 for n in (graph_replays,wall_calls)):
        raise ValueError('at least four rounds and positive iteration counts required')
    if type(recompute) is not bool or type(large_tiles) is not bool:raise ValueError('candidate opt-ins must be boolean')
    if type(reuse_calibration_graphs) is not bool:raise ValueError('reuse_calibration_graphs must be boolean')
    candidates(recompute=recompute,large_tiles=large_tiles,segmented=segmented,merge_statistics=merge_statistics)
    options=dict(compile_options or {})
    if any(k in options for k in ('shape_bindings','cross_domain_strategy','platform_config')):
        raise ValueError('candidate configurations and symbolic bindings are owned by strategy tuning')
    if options.get('autotune') or options.get('autotune_fold') or options.get('custom_ops') or options.get('concurrency','sequential')!='sequential':
        raise ValueError('strategy tuning requires fixed native sequential candidates')
    if options.get('gpu_backend','triton')!='triton' or not options.get('index_fusion',True) or not options.get('reduction_fusion',True):
        raise ValueError('strategy tuning requires native indexed/reduction fusion')
    return dict(schema_version=1,execution_surface='prepared_codegen',source_sha256=_sha(source),
                compiler_fingerprint=compiler_fingerprint(),pipeline=pipeline,bindings=bindings,
                input_context=context,device=device,objective=objective,workload_tag=workload_tag,
                recompute=recompute,large_tiles=large_tiles,segmented=segmented,merge_statistics=merge_statistics,compile_options=options,min_gain=min_gain,
                max_round_regression=max_round_regression,measurement=dict(rounds=rounds,graph_replays=graph_replays,
                wall_calls=wall_calls,reuse_calibration_graphs=reuse_calibration_graphs,
                retained_graph_budget_bytes=GRAPH_RETENTION_BUDGET_BYTES,
                wall_definition='batch synchronized mean, not individual-call completion latency'))


def bound_comparison(source, identity, evidence):
    """Reject stale host/timing evidence before passing observations to analysis."""
    if _sha(source)!=identity['source_sha256'] or compiler_fingerprint()!=identity['compiler_fingerprint']:
        raise ValueError('policy source/compiler changed')
    if any(evidence.get(k)!=identity[k] for k in ('source_sha256','compiler_fingerprint','device')):
        raise ValueError('evidence source/compiler/device mismatch')
    expected=candidates(recompute=identity['recompute'],large_tiles=identity['large_tiles'],segmented=identity.get('segmented',False),merge_statistics=identity.get('merge_statistics',False))
    if identity['measurement'].get('reuse_calibration_graphs'):
        stats=evidence.get('calibration_capture',{})
        n,r=len(expected),identity['measurement']['rounds']
        budget=identity['measurement']['retained_graph_budget_bytes']
        if (stats.get('requested_reuse') is not True or stats.get('retention_budget_bytes')!=budget
                or type(stats.get('retained_bytes_peak')) is not int or stats['retained_bytes_peak']<0
                or type(stats.get('captures')) is not int):
            raise ValueError('missing or invalid calibration capture evidence')
        reused=(stats.get('mode')=='per_calibration' and stats['captures']==n
                and stats['retained_bytes_peak']<=budget and stats.get('fallback_reason') is None)
        fallback=(stats.get('mode')=='per_round' and n*r<stats['captures']<=n*(r+1)
                  and stats['retained_bytes_peak']>budget
                  and stats.get('fallback_reason')=='retained allocation budget exceeded')
        if not (reused or fallback):raise ValueError('inconsistent calibration capture evidence')
    records=evidence.get('records',[])
    if not isinstance(records,list) or len(records)!=len(expected):raise ValueError('one observation per candidate required')
    for spec,record in zip(expected,records):
        if record.get('candidate')!=spec.name or record.get('bindings')!=identity['bindings'] or record.get('input_context')!=identity['input_context']:
            raise ValueError('candidate/context ordering mismatch')
    records_by_name={record['candidate']:record for record in records}
    def verify_deployment(spec,p):
        record=records_by_name[spec.name]
        names=[n.output for n in p.dataflow.nodes if n.op=='input']
        proof=_prepared_proof(module([p.triton]),identity['pipeline'],identity['input_context'],identity['device'],names)
        if record.get('deployment')!=proof or record.get('module_sha256')!=proof['original_module_sha256']:
            raise ValueError('prepared execution source mismatch')
    comparison=_compare_strategies(source,identity['bindings'],recompute=identity['recompute'],large_tiles=identity['large_tiles'],segmented=identity.get('segmented',False),merge_statistics=identity.get('merge_statistics',False),
        compile_options=identity['compile_options'],sm_count=identity['device']['sm_count'],evidence=evidence,
        inspect_candidate=verify_deployment)
    comparison['execution_surface']='prepared_codegen'
    for item,record in zip(comparison['candidates'],records):
        item['prepared_source_sha256']=record['deployment']['source_sha256']
        if item['evidence_status']!='measured_for_recorded_context':
            raise ValueError('stale or incomplete observations: '+item['name']+' '+str(item['evidence_rejections']))
        for metric in ('graph_us','wall_us'):
            if len(item['observation']['samples'][metric])!=identity['measurement']['rounds']:
                raise ValueError('paired round count mismatch')
        item['observation']['execution_surface']='prepared_codegen'
        item['decision_features']=dict(launches=item['launches'],logical_traffic_bytes=item['logical_traffic_bytes'],
            intermediate_bytes=item['intermediate_bytes'],cross_domain_applied=item['cross_domain_applied'],
            kernels=item['kernels'])
        if any(k['resources'][key] is None for k in item['kernels']
               for key in ('n_regs','n_spills_reported','shared_reported','num_warps')):
            item['selection_rejection']='compiled resource evidence incomplete; retain materialize'
        if any(k['resources']['num_warps'] is not None and (type(k['resources']['num_warps']) is not int
               or k['resources']['num_warps']<=0) for k in item['kernels']):
            raise ValueError('invalid warp resource observation')
        if item['mode']=='segmented-merge' and not item['merged_statistics_applied']:
            item['selection_rejection']='merged statistics lowering was not applied'
        if item['mode']=='segmented' and not item['segmented_applied']:
            item['selection_rejection']='segmented stage lowering was not applied'
        if item['mode']=='recompute' and not item['cross_domain_applied']:
            item['selection_rejection']='cross-domain recomputation was not applied'
    if comparison['candidates'][0].get('selection_rejection'):
        for item in comparison['candidates']:
            item['selection_rejection']='default resource evidence incomplete; retain materialize'
    return comparison


def load_policy(path, source, identity):
    if not Path(path).exists():return None,'miss'
    try:
        entry=json.loads(Path(path).read_text());payload=entry['payload']
        if entry['sha256']!=digest(payload) or payload['identity']!=identity:return None,'identity or digest mismatch'
        comparison=bound_comparison(source,identity,payload['evidence'])
        selected=select_paired(comparison,identity['objective'],min_gain=identity['min_gain'],max_round_regression=identity['max_round_regression'])
        if selected!=payload['selection']:return None,'selection mismatch'
        return payload,'hit'
    except (OSError,ValueError,KeyError,TypeError,StopIteration):return None,'invalid or stale policy'


def tune_strategy_pipeline(source, *, pipeline, args, cache_dir, objective, workload_tag, validator,
                           recompute=False, large_tiles=False, segmented=False, merge_statistics=False, compile_options=None,
                           min_gain=.05, max_round_regression=.05, rounds=6, graph_replays=200, wall_calls=50,
                           reuse_calibration_graphs=False):
    """Return a prepared function and explainable exact-context empirical policy.

    Calibration and code/JIT preparation occur outside capture, once per context.
    A policy hit rebinds source/resources/samples and validates new outputs, but
    does not time candidates again. The returned function adds no dispatch wrapper.
    """
    import torch
    import triton
    from compiler.compile import compile_source
    from compiler.runtime.runtime import prepare_pipeline
    start=time.perf_counter()
    ledger=PhaseLedger()
    if torch.cuda.is_current_stream_capturing():raise ValueError('tune outside CUDA Graph capture')
    if not callable(validator):raise ValueError('an independent validator is required')
    options=dict(compile_options or {})
    policy_identity(source,pipeline,{}, {}, {},objective=objective,workload_tag=workload_tag,
        recompute=recompute,large_tiles=large_tiles,segmented=segmented,merge_statistics=merge_statistics,compile_options=options,min_gain=min_gain,max_round_regression=max_round_regression,
        rounds=rounds,graph_replays=graph_replays,wall_calls=wall_calls,reuse_calibration_graphs=reuse_calibration_graphs)
    compiled=ledger.call('source_analysis',compile_source,source,**options)
    if len(compiled.pipelines)!=1 or compiled.pipelines[0].pipeline.name!=pipeline:
        raise ValueError('one named pipeline required')
    p=compiled.pipelines[0]
    if any(k.kind not in {'primitive','pointwise'} for k in p.triton.kernels):
        raise ValueError('strategy tuning currently supports pure compositional reductions')
    specs=candidates(recompute=recompute,large_tiles=large_tiles,segmented=segmented,merge_statistics=merge_statistics)
    runs={};preparations={}
    default=specs[0]
    runs[default.name],preparations[default.name]=ledger.call('prepare_default',prepare_pipeline,source,pipeline=pipeline,args=args,
        cache_dir=Path(cache_dir)/'deployments',validator=validator,**options,**default.options())
    names=[n.output for n in p.dataflow.nodes if n.op=='input']
    native=preparations[default.name]['identity']['context']
    context={name:dict(dtype=p.dataflow.values[name].dtype,shape=c['shape'],stride=c['stride'],alignment_mod128=c['alignment_mod128']) for name,c in zip(names,native)}
    bindings={}
    for name,c in context.items():
        for d,size in zip(p.dataflow.values[name].shape,c['shape']):
            if not str(d).isdigit():bindings[str(d)]=size
    props=torch.cuda.get_device_properties(args[0].device)
    device=dict(name=props.name,uuid=str(props.uuid),total_memory=props.total_memory,sm_count=props.multi_processor_count,
                capability=[props.major,props.minor],index=args[0].device.index,torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__)
    identity=policy_identity(source,pipeline,bindings,context,device,objective=objective,workload_tag=workload_tag,
        recompute=recompute,large_tiles=large_tiles,segmented=segmented,merge_statistics=merge_statistics,compile_options=options,min_gain=min_gain,max_round_regression=max_round_regression,
        rounds=rounds,graph_replays=graph_replays,wall_calls=wall_calls,reuse_calibration_graphs=reuse_calibration_graphs)
    path=Path(cache_dir)/'policies'/(digest(identity)+'.json')
    payload,status=ledger.call('policy_lookup_revalidation',load_policy,path,source,identity)
    if payload is None:
        # A hit already reconstructs every candidate, its host proof and
        # features in load_policy/bound_comparison. Only misses need this
        # separate legality pass before any new resource probing/calibration.
        ledger.call('legality_features',compare_strategies,source,bindings,recompute=recompute,large_tiles=large_tiles,segmented=segmented,merge_statistics=merge_statistics,sm_count=device['sm_count'],compile_options=options)
        evidence=dict(schema_version=1,source_sha256=identity['source_sha256'],compiler_fingerprint=identity['compiler_fingerprint'],device=device,records=[])
        for spec in specs:
            if spec.name not in runs:
                runs[spec.name],preparations[spec.name]=ledger.call('prepare_candidate',prepare_pipeline,source,pipeline=pipeline,args=args,
                    cache_dir=Path(cache_dir)/'deployments',validator=validator,**options,**spec.options())
            run=runs[spec.name];result,resources=ledger.call('resource_probe',probe_resources,run,args)
            if validator(result) is not True:raise ValueError('candidate independent validation failed: '+spec.name)
            prep=preparations[spec.name]
            proof={k:prep['identity'][k] for k in ('original_module_sha256','module_sha256','context','device_key')}
            proof.update(source_sha256=prep['source_sha256'],selections=prep['selections'])
            evidence['records'].append(dict(candidate=spec.name,bindings=bindings,input_context=context,
                module_sha256=proof['original_module_sha256'],deployment=proof,kernels=resources,correctness_passed=True,split='exact_context_calibration'))
        calls={name:lambda run=run:run(*args) for name,run in runs.items()}
        capture_stats={}
        samples=ledger.call('paired_calibration',measure_calls,calls,rounds=rounds,graph_replays=graph_replays,wall_calls=wall_calls,
                           reuse_graphs=reuse_calibration_graphs,capture_stats=capture_stats)
        evidence['calibration_capture']=capture_stats
        for record in evidence['records']:record['samples']=samples[record['candidate']]
        comparison=ledger.call('evidence_reconstruction',bound_comparison,source,identity,evidence)
        selection=ledger.call('paired_selection',select_paired,comparison,objective,min_gain=min_gain,max_round_regression=max_round_regression)
        payload=dict(identity=identity,selection=selection,evidence=evidence)
        if compiler_fingerprint()!=identity['compiler_fingerprint']:raise ValueError('compiler changed during calibration')
        ledger.call('policy_record_write',_atomic_json,path,dict(payload=payload,sha256=digest(payload)))
    selected=payload['selection']['selected']
    if selected not in runs:
        spec=next(s for s in specs if s.name==selected)
        runs[selected],preparations[selected]=ledger.call('prepare_selected_cache_hit',prepare_pipeline,source,pipeline=pipeline,args=args,
            cache_dir=Path(cache_dir)/'deployments',validator=validator,**options,**spec.options())
    if compiler_fingerprint()!=identity['compiler_fingerprint']:raise ValueError('compiler changed during preparation')
    total=(time.perf_counter()-start)*1000
    return runs[selected],dict(identity=identity,selection=payload['selection'],cache_status=status,policy_path=str(path),
        preparation=preparations[selected],preparation_ms=total,phase_costs=ledger.finish(total),
        candidate_preparation_costs={s.name:(dict(preparation_ms=preparations[s.name]['preparation_ms'],
            cache_hit=preparations[s.name]['cache_hit']) if s.name in preparations else None) for s in specs},
        calibration_capture=payload['evidence'].get('calibration_capture'),
        limitations=['same-context empirical selection; new shape/stride/device requires preparation',
                     'compiled resources are not occupancy or profiler HBM counters',
                     'wall_us is batch throughput; no individual-call completion-latency claim'])
