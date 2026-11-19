"""Explicit, exact-context paired tuning. Never changes compiler defaults.

Source and policy caches are separate. A policy hit must revalidate all emitted
modules and bound measurement records. Input contents are not cached; callers
name the workload distribution. Metadata guards do not prove equal performance
for arbitrary new index values. Tuning is forbidden during graph capture.
"""
import hashlib
import json
from pathlib import Path

from compiler.shared import compiler_fingerprint
from compiler.optimization.measurement import (digest, select_paired, _atomic_json, probe_resources, measure_calls)
from compiler.optimization.gather_strategy import compare_gather_strategies, candidates


def policy_identity(source, bindings, device, context, objective, workload_tag, min_gain, max_round_regression):
    if not isinstance(workload_tag, str) or not workload_tag.strip():
        raise ValueError('an explicit workload_tag is required')
    return dict(schema_version=1, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                compiler_fingerprint=compiler_fingerprint(), bindings=bindings,
                device=device, input_context=context, objective=objective,
                workload_tag=workload_tag, min_gain=min_gain, max_round_regression=max_round_regression,
                candidate_space='row-gather-partial-v1')


def save_policy(cache_dir, identity, comparison, evidence):
    if any(i['observation'] and i['observation']['input_context'] != identity['input_context']
           for i in comparison['candidates']):
        raise ValueError('policy input context mismatch')
    selection = select_paired(comparison, identity['objective'], min_gain=identity['min_gain'],
                              max_round_regression=identity['max_round_regression'])
    payload = dict(identity=identity, selection=selection, evidence=evidence)
    path = Path(cache_dir) / (digest(identity)+'.json')
    _atomic_json(path, dict(payload=payload, sha256=digest(payload)))
    return selection, path


def load_policy(cache_dir, identity, source):
    path = Path(cache_dir) / (digest(identity)+'.json')
    if not path.exists():return None, 'miss'
    try:
        entry = json.loads(path.read_text());payload = entry['payload']
        if entry['sha256'] != digest(payload) or payload['identity'] != identity:
            return None, 'corrupt identity or digest'
        evidence = payload['evidence']
        if evidence.get('device') != identity['device']:
            return None, 'device mismatch'
        report = compare_gather_strategies(source, identity['bindings'],
                    sm_count=identity['device']['sm_count'], evidence=evidence, partial=True)
        for item in report['candidates']:
            if item['evidence_status'] != 'measured_for_recorded_context':
                return None, 'missing or stale candidate evidence'
            if item['observation']['input_context'] != identity['input_context']:
                return None, 'layout mismatch'
        selected = select_paired(report, identity['objective'], min_gain=identity['min_gain'],
                                 max_round_regression=identity['max_round_regression'])
        if selected != payload['selection']:return None, 'selection mismatch'
        return selected, 'hit'
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return None, 'invalid policy entry'


def tune_gather_pipeline(source, *, pipeline, args, cache_dir, objective, workload_tag,
                         validator, min_gain=.05, max_round_regression=.05):
    """Prepare outside graph capture. Return guarded selected run + policy record.

    The required validator checks each generated result independently. The returned
    run is guarded to the exact tuned metadata; new contexts require preparation.
    Cold compilation/tuning is reported separately and never hidden in hot latency.
    """
    import torch
    import triton
    import time
    from compiler.compile import compile_source
    from compiler.runtime.runtime import compile_pipeline
    if torch.cuda.is_current_stream_capturing():
        raise ValueError('prepare/tune must run outside CUDA Graph capture')
    if not callable(validator):raise ValueError('an independent validator is required')
    p=compile_source(source).pipelines[0]
    if len(compile_source(source).pipelines)!=1 or p.pipeline.name!=pipeline:
        raise ValueError('tuning requires one named gather pipeline')
    inputs=[n for n in p.dataflow.nodes if n.op=='input']
    if len(inputs)!=len(args) or any(a.device.type!='cuda' for a in args):
        raise ValueError('tuning requires all declared CUDA input tensors')
    dev=args[0].device
    if any(a.device!=dev for a in args):raise ValueError('one CUDA device required')
    if dev.index != torch.cuda.current_device():
        raise ValueError('prepare inside torch.cuda.device(input.device)')
    inverse={torch.float16:'fp16',torch.bfloat16:'bf16',torch.float32:'fp32',torch.int32:'int32',torch.int64:'int64',torch.bool:'bool'}
    bindings={};context={}
    for node,a in zip(inputs,args):
        t=p.dataflow.values[node.output]
        if len(a.shape)!=len(t.shape) or inverse.get(a.dtype)!=t.dtype:raise ValueError('input type mismatch')
        for d,size in zip(t.shape,a.shape):
            if size<=0:raise ValueError('tuning requires positive dimensions')
            if str(d).isdigit():
                if int(d)!=size:raise ValueError('static dimension mismatch')
            elif d in bindings and bindings[d]!=size:raise ValueError('symbolic dimension mismatch')
            else:bindings[d]=size
        context[node.output]=dict(dtype=inverse[a.dtype],shape=list(a.shape),stride=list(a.stride()),alignment_mod128=a.data_ptr()%128)
    props=torch.cuda.get_device_properties(dev)
    device=dict(name=props.name,uuid=str(props.uuid),sm_count=props.multi_processor_count,
                capability=[props.major,props.minor],torch=torch.__version__,triton=triton.__version__)
    identity=policy_identity(source,bindings,device,context,objective,workload_tag,min_gain,max_round_regression)
    start=time.perf_counter();selected,status=load_policy(Path(cache_dir)/'policies',identity,source)
    options=candidates(source,bindings,partial=True)
    if selected is None:
        calls={};resources={}
        for name,option in options.items():
            run=compile_pipeline(source,pipeline=pipeline,cache_dir=Path(cache_dir)/'modules',**option)
            result,resources[name]=probe_resources(run,args)
            if validator(result) is not True:raise ValueError('independent correctness check failed: '+name)
            calls[name]=lambda run=run:run(*args)
        with torch.cuda.device(dev):samples=measure_calls(calls)
        structural=compare_gather_strategies(source,bindings,sm_count=props.multi_processor_count,partial=True)
        evidence=dict(schema_version=1,source_sha256=identity['source_sha256'],compiler_fingerprint=identity['compiler_fingerprint'],device=device,records=[])
        for item in structural['candidates']:
            name=item['name'];evidence['records'].append(dict(candidate=name,bindings=bindings,
                module_sha256=item['module_sha256'],input_context=context,kernels=resources[name],samples=samples[name],correctness_passed=True,split='exact_context_tuning'))
        measured=compare_gather_strategies(source,bindings,sm_count=props.multi_processor_count,evidence=evidence,partial=True)
        selected,path=save_policy(Path(cache_dir)/'policies',identity,measured,evidence)
    else:path=Path(cache_dir)/'policies'/(digest(identity)+'.json')
    if compiler_fingerprint()!=identity['compiler_fingerprint']:
        raise ValueError('compiler changed during tuning')
    run=compile_pipeline(source,pipeline=pipeline,cache_dir=Path(cache_dir)/'modules',**options[selected['selected']])
    def guarded(*new_args):
        if len(new_args)!=len(args) or any(a.device!=dev or inverse.get(a.dtype)!=c['dtype']
            or list(a.shape)!=c['shape'] or list(a.stride())!=c['stride'] or a.data_ptr()%128!=c['alignment_mod128']
            for a,c in zip(new_args,context.values())):
            raise ValueError('tuned gather context changed; prepare a new policy')
        return run(*new_args)
    return guarded,dict(cache_status=status,selection=selected,policy_path=str(path),
                        preparation_ms=(time.perf_counter()-start)*1000,identity=identity)
