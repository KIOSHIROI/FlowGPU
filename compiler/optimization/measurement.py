"""Shared empirical selection, paired measurement and resource capture.

CUDA calls are imported lazily. Structural features explain decisions; timings
select only within the recorded context, never predict unseen-shape latency.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

from compiler.analysis import power as _power

GRAPH_RETENTION_BUDGET_BYTES=64*1024*1024

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def select_paired(comparison, objective, *, min_gain=.05, max_round_regression=.05,
                  noise_band=None):
    """Empirical paired-round gate, not a confidence interval or causal model.

    ``noise_band`` optionally raises the bar to the declared measurement noise:
    a mapping with ``sigma`` (paired log-ratio standard deviation from an
    earlier run) and ``rounds``. It must be declared before the comparison is
    read, because deriving the band from the same data and then testing against
    it is circular. Without it the gate behaves exactly as before.
    """
    if objective not in {'graph_us', 'wall_us'}:
        raise ValueError('objective must be graph_us or wall_us')
    if not 0 <= min_gain < 1 or not 0 <= max_round_regression < 1:
        raise ValueError('gain and round regression limits must be in [0,1)')
    band = None
    if noise_band is not None:
        if not isinstance(noise_band, dict) or 'sigma' not in noise_band or 'rounds' not in noise_band:
            raise ValueError('noise_band requires sigma and rounds')
        band = _power.report(noise_band['sigma'], noise_band['rounds'])
        band['applied'] = True
        band['declared'] = True
    floor = min_gain if band is None else max(min_gain, band['mde_relative'])
    baseline = next(i for i in comparison['candidates'] if i['name'] == comparison['default'])
    def samples(item):
        if item['evidence_status'] != 'measured_for_recorded_context':
            return None
        values = item['observation'].get('samples',{}).get(objective)
        if (not isinstance(values,list) or not values or any(type(v) not in {int,float}
                or not math.isfinite(v) or v<=0 for v in values)):
            return None
        return values
    control = samples(baseline)
    if control is None or len(control) < 4:
        raise ValueError('paired tuning requires at least four validated default rounds')
    if band is not None and len(control) < noise_band['rounds']:
        raise ValueError('noise_band declared {} rounds but the comparison collected {}; '
                         'a weaker design than declared cannot use its band'.format(
                             noise_band['rounds'], len(control)))
    decisions = []
    for item in comparison['candidates']:
        values = samples(item)
        reason = 'no matched measurements'
        ratio = worst = None
        if item.get('selection_rejection'):
            reason = item['selection_rejection']
        elif values is not None and len(values) == len(control):
            ratio = statistics.median(values) / statistics.median(control)
            worst = max(v / b for v, b in zip(values, control))
            if ratio <= 1-min_gain and worst <= 1+max_round_regression:
                reason = 'eligible'
            else:
                reason = 'insufficient median gain or unstable paired rounds'
            if reason == 'eligible' and band is not None and (1 - ratio) <= band['mde_relative']:
                reason = 'within declared measurement noise'
        decisions.append(dict(candidate=item['name'], median_ratio=ratio,
                              worst_paired_round_ratio=worst, reason=reason,
                              features=item['decision_features']))
    eligible = [d for d in decisions if d['reason'] == 'eligible']
    selected = min(eligible, key=lambda d:d['median_ratio'])['candidate'] if eligible else comparison['default']
    # Paired noise is estimated from the same ratio series, so it describes this
    # comparison rather than a population. Unknown noise stays None: treating it
    # as zero is how an underpowered run reads as a win.
    series=[(values,control) for item in comparison['candidates']
            if (values:=samples(item)) is not None and len(values)==len(control)]
    sigma=_power.pooled_log_ratio_sigma(series)
    scales=_power.paired_noise_scales(series)
    return dict(selected=selected, objective=objective, min_gain=min_gain,
                max_round_regression=max_round_regression, decisions=decisions,
                effective_gain_floor=floor, noise_band=band,
                power=(None if sigma is None else _power.report(
                    sigma, len(control), candidates=len(comparison['candidates']),
                    robust_sigma=scales['robust'], outliers=scales['outliers'])),
                note='same-context empirical selection; not an unseen-shape prediction')


def require_detectable(protocol, *, alpha=0.05, power=0.8):
    """Refuse to start a run whose target effect the design cannot resolve.

    The protocol must declare three things, because none of them can be
    recovered after the fact:

    ``expected_effect``  the relative improvement the run is meant to detect
    ``noise_sigma``      the paired log-ratio standard deviation from a
                         comparable prior run on the same objective
    ``rounds``           paired rounds the run will collect

    A missing field is an error rather than a default. Assuming "no noise" is
    how an underpowered run ends up looking like a success: the winner of k
    noisy candidates is optimistic by roughly E[max of k] * sigma, which for
    seven candidates is about 1.35 sigma.

    Returns the report dict to embed in the result file; raises ValueError with
    the required round count when the design is too weak.
    """
    if not isinstance(protocol, dict):
        raise ValueError('protocol must be a mapping')
    missing = [k for k in ('expected_effect', 'noise_sigma', 'rounds') if k not in protocol]
    if missing:
        raise ValueError('protocol must declare ' + ', '.join(missing))
    effect = protocol['expected_effect']
    sigma = protocol['noise_sigma']
    rounds = protocol['rounds']
    objective = protocol.get('objective')
    if isinstance(sigma, dict):
        # Graph and full-call timing do not share a noise floor: on the H100
        # probe graph sigma was 0.06-0.37% while wall sigma was 0.68-0.84%.
        # Applying one number to both is either too lax or needlessly strict.
        if objective not in sigma:
            raise ValueError('noise_sigma names objectives ' + ', '.join(sorted(sigma))
                             + '; protocol must select one with "objective"')
        sigma = sigma[objective]
    if type(rounds) is not int or rounds < 2:
        raise ValueError('rounds must be an integer of at least two')
    if isinstance(effect, bool) or not isinstance(effect, (int, float)) or not 0 < effect < 1:
        raise ValueError('expected_effect must be a fraction in (0,1)')
    candidates = protocol.get('candidates', protocol.get('blocks') and len(protocol['blocks']))
    summary = _power.report(sigma, rounds, effect, candidates=candidates, alpha=alpha, power=power)
    if objective is not None:
        summary['objective'] = objective
    if not summary['detectable']:
        raise ValueError(
            'expected_effect {} is below the minimum detectable effect {} at sigma {} with {} '
            'paired rounds; {} rounds would be needed'.format(
                effect, round(summary['mde_relative'], 4), sigma, rounds,
                summary['required_rounds']))
    return summary


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def probe_resources(run, args, *, artifact_dir=None, label=None):
    """Record actual JIT specialization resources/IR outside timing, then restore."""
    import triton
    originals=[];captured=[];seen=set()
    def wrap(original,name):
        def launch(*a,**kw):
            compiled=original(*a,**kw)
            if compiled is not None and hasattr(compiled,'metadata'):
                captured.append(dict(name=name, n_regs=getattr(compiled,'n_regs',None),
                    n_spills_reported=getattr(compiled,'n_spills',None),
                    shared_reported=getattr(compiled.metadata,'shared',None),
                    num_warps=getattr(compiled.metadata,'num_warps',None)))
                if artifact_dir:
                    dest=Path(artifact_dir)/label;dest.mkdir(parents=True,exist_ok=True)
                    for kind in ['ttir','ttgir','llir','ptx']:
                        content=compiled.asm.get(kind)
                        if isinstance(content,str):(dest/(name+'.'+kind)).write_text(content)
            return compiled
        return launch
    for name,fn in list(run.__globals__.items()):
        if isinstance(fn,triton.runtime.JITFunction) and id(fn) not in seen:
            seen.add(id(fn));originals.append((fn,'run' in fn.__dict__,fn.__dict__.get('run')))
            fn.run=wrap(fn.run,name)
    try:result=run(*args)
    finally:
        for fn,had,value in originals:
            if had:fn.run=value
            else:delattr(fn,'run')
    return result,captured


def measure_calls(calls, *, rounds=6, graph_replays=200, wall_calls=50,
                  reuse_graphs=False, capture_stats=None):
    """Graph event mean and batch-synchronized full-call throughput, paired order."""
    if type(rounds) is not int or rounds < 4:raise ValueError('at least four paired rounds required')
    if any(type(n) is not int or n<=0 for n in (graph_replays,wall_calls)):
        raise ValueError('positive graph/wall iteration counts required')
    if type(reuse_graphs) is not bool:raise ValueError('reuse_graphs must be boolean')
    if capture_stats is not None and type(capture_stats) is not dict:raise ValueError('capture_stats must be a dict')
    import torch
    import time
    stats=dict(requested_reuse=reuse_graphs,captures=0,retained_bytes_peak=0,
               retention_budget_bytes=GRAPH_RETENTION_BUDGET_BYTES,mode='per_round',fallback_reason=None)
    if capture_stats is not None:
        capture_stats.clear();capture_stats.update(stats);stats=capture_stats
    samples={name:dict(graph_us=[],wall_us=[]) for name in calls}
    graphs={}
    def capture(call):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):keep=call()
        stats['captures']+=1
        return graph,keep
    for call in calls.values():
        for _ in range(10):call()
    torch.cuda.synchronize()
    graph=keep=None
    try:
        if reuse_graphs:
            baseline=torch.cuda.memory_allocated()
            for name,call in calls.items():
                graphs[name]=capture(call)
                delta=max(0,torch.cuda.memory_allocated()-baseline)
                stats['retained_bytes_peak']=max(stats['retained_bytes_peak'],delta)
                if delta>stats['retention_budget_bytes']:
                    # Revert the measurement mechanism, never omit candidates.
                    # Runtime/kernel failures are propagated rather than hidden.
                    torch.cuda.synchronize();graphs.clear()
                    stats['fallback_reason']='retained allocation budget exceeded'
                    break
            else:stats['mode']='per_calibration'
        for iteration in range(rounds):
            for name in list(calls) if iteration%2==0 else list(reversed(calls)):
                call=calls[name]
                for _ in range(5):call()
                torch.cuda.synchronize()
                graph,keep=graphs[name] if graphs else capture(call)
                for _ in range(10):graph.replay()
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(graph_replays):graph.replay()
                end.record();end.synchronize()
                samples[name]['graph_us'].append(start.elapsed_time(end)*1000/graph_replays)
                graph=keep=None
                torch.cuda.synchronize();start_wall=time.perf_counter()
                for _ in range(wall_calls):call()
                torch.cuda.synchronize()
                samples[name]['wall_us'].append((time.perf_counter()-start_wall)*1e6/wall_calls)
    finally:
        graphs.clear();graph=keep=None
    return samples
