"""Experimental candidate-wise cost ranking. Never changes lowering defaults.

Fits log latency ratios, not physical bandwidth/occupancy. Resource interactions
are explanatory proxies. Shape-grouped validation and conservative abstention
do not constitute calibrated statistical confidence intervals.
"""
import math
import statistics

FEATURES = ('launches', 'log_traffic', 'log_scan_pressure', 'log_grid_work',
            'log_tile', 'log_padding', 'log_regs', 'log_shared',
            'log_spill', 'log_spill_scan_pressure')
RIDGES = (.1, 1., 10., 100.)


def candidate_cost_features(item, sm_count):
    """Use each emitted launch and its own compiled resources, not one probe."""
    if type(sm_count) is not int or sm_count <= 0:
        raise ValueError('candidate costs require actual positive SM count')
    if item['evidence_status'] not in {'compiled_resources_for_recorded_context', 'measured_for_recorded_context'}:
        raise ValueError('candidate costs require bound resource evidence')
    kernels = item['kernels']
    if not kernels:
        raise ValueError('candidate has no kernels')
    resources = [k.get('resources') or {} for k in kernels]
    keys = ('n_regs', 'shared_reported', 'n_spills_reported')
    if any(type(r.get(key)) is not int or r[key] < 0 for r in resources for key in keys):
        raise ValueError('required compiled resource field missing or invalid')
    pressure = [k['chunk_iterations_per_program'] / (min(k['programs'], sm_count) / sm_count)
                for k in kernels]
    raw = (item['launches'], item['logical_traffic_bytes'], sum(pressure),
           sum(k['program_chunk_iterations'] for k in kernels),
           max(k['tile_elements'] or 0 for k in kernels),
           max(1 / k['useful_lane_fraction'] for k in kernels),
           max(r['n_regs'] for r in resources), max(r['shared_reported'] for r in resources),
           sum(r['n_spills_reported'] for r in resources),
           sum(r['n_spills_reported'] * p for r, p in zip(resources, pressure)))
    values = dict(zip(FEATURES, [raw[0], *(math.log1p(v) for v in raw[1:])]))
    if any(not math.isfinite(v) for v in values.values()):
        raise ValueError('non-finite candidate feature')
    return values


def _context(comparison):
    observations = [i.get('observation') for i in comparison['candidates']]
    if any(not o for o in observations):
        raise ValueError('all candidate contexts must be bound')
    contexts = [dict(device=o['device'], input_dtypes={n:s['dtype'] for n,s in o['input_context'].items()})
                for o in observations]
    if any(c != contexts[0] for c in contexts):
        raise ValueError('candidate device/dtype contexts disagree')
    return contexts[0]


def _vectors(comparison):
    control = next(i for i in comparison['candidates'] if i['name'] == 'materialize')
    base = candidate_cost_features(control, comparison['sm_count'])
    return {i['name']:[v-base[f] for f,v in candidate_cost_features(i,comparison['sm_count']).items()]
            for i in comparison['candidates']}


def _layout(comparison):
    def signature(context):
        return {name:dict(order=sorted((j for j,s in enumerate(spec['shape']) if s>1),
                                       key=lambda j:(-spec['stride'][j],j)),
                          broadcast=[j for j,s in enumerate(spec['stride']) if s==0])
                for name,spec in context.items()}
    layouts = [signature(i['observation']['input_context']) for i in comparison['candidates']]
    if any(v != layouts[0] for v in layouts):
        raise ValueError('candidate input layouts disagree')
    return layouts[0]


def _solve(matrix, target):
    """Small positive-definite ridge system; no optional numerical dependency."""
    rows = [list(r)+[v] for r,v in zip(matrix,target)]
    n = len(target)
    for j in range(n):
        pivot = max(range(j,n), key=lambda i:abs(rows[i][j]))
        rows[j],rows[pivot] = rows[pivot],rows[j]
        scale = rows[j][j]
        if abs(scale) < 1e-12:
            raise ValueError('singular cost fit')
        rows[j] = [v/scale for v in rows[j]]
        for i in range(n):
            if i != j:
                factor = rows[i][j]
                rows[i] = [a-factor*b for a,b in zip(rows[i],rows[j])]
    return [r[-1] for r in rows]


def _fit(points, ridge):
    n = len(FEATURES); weight = sum(p[2] for p in points)
    scales = [max(math.sqrt(sum(w*x[j]**2 for x,y,w in points)/weight),1e-8) for j in range(n)]
    scaled = [([v/s for v,s in zip(x,scales)],y,w/weight) for x,y,w in points]
    matrix = [[sum(w*x[j]*x[k] for x,y,w in scaled)+(ridge if j==k else 0) for k in range(n)] for j in range(n)]
    target = [sum(w*x[j]*y for x,y,w in scaled) for j in range(n)]
    coefficients = [v/s for v,s in zip(_solve(matrix,target),scales)]
    return coefficients


def _dot(a,b):
    return sum(x*y for x,y in zip(a,b))


def fit_candidate_policy(rows, metric, *, min_advantage=.05, error_quantile=.9):
    """Screen-only, shape-grouped ridge selection with per-candidate error gates.

    Coefficients are shared across candidates; input vectors are candidate minus
    materialize. No R threshold or holdout timings participate. Validation uses
    equally weighted shapes (optional workload weights govern coefficient fit).
    """
    if metric not in {'graph_us','wall_us'} or len(rows)<4:
        raise ValueError('known objective and at least four screen shapes required')
    if not 0 <= min_advantage < 1 or not 0 < error_quantile <= 1:
        raise ValueError('invalid conservative gate parameters')
    first = rows[0]['comparison']; names = [i['name'] for i in first['candidates']]
    context = _context(first)
    seen = set(); groups = []; vectors = []; errors_by_ridge = {}; layouts = []
    for row in rows:
        c = row['comparison']; shape = tuple(sorted(c['bindings'].items()))
        if shape in seen:
            raise ValueError('screen shapes must be distinct')
        seen.add(shape)
        if row.get('split') != 'screen' or [i['name'] for i in c['candidates']] != names:
            raise ValueError('identical candidates and screening rows required')
        if any(c[k] != first[k] for k in ['source_sha256','compiler_fingerprint','sm_count']):
            raise ValueError('screen source/flowgpu/compiler/SM identity mismatch')
        if _context(c) != context:
            raise ValueError('screen device/dtype identity mismatch')
        layout = _layout(c)
        if layout not in layouts:
            layouts.append(layout)
        if [i['module_sha256'] for i in c['candidates']] != [i['module_sha256'] for i in first['candidates']]:
            raise ValueError('screen generated module identity mismatch')
        weight = row.get('weight',1)
        if type(weight) not in {int,float} or not math.isfinite(weight) or weight <= 0:
            raise ValueError('positive finite screen weight required')
        for i in c['candidates']:
            samples = i['observation'].get('samples',{}).get(metric,[])
            measured = row['medians'][i['name']][metric]
            if not samples or statistics.median(samples) != measured or not math.isfinite(measured) or measured<=0:
                raise ValueError('training medians must match bound measurements')
        v = _vectors(c); vectors.append(v)
        groups.append([(v[name],math.log(row['medians'][name][metric]/row['medians']['materialize'][metric]),weight)
                       for name in names if name!='materialize'])
    for ridge in RIDGES:
        errors = {name:[] for name in names if name!='materialize'}
        for j,group in enumerate(groups):
            coefficients = _fit([p for k,g in enumerate(groups) if k!=j for p in g],ridge)
            for name,(x,y,w) in zip(errors,group):
                errors[name].append(y-_dot(x,coefficients))
        errors_by_ridge[ridge] = errors
    def score(ridge):
        e = [abs(v) for values in errors_by_ridge[ridge].values() for v in values]
        return statistics.mean(e)+.25*max(e)
    ridge = min(RIDGES,key=lambda r:(score(r),-r))
    errors = errors_by_ridge[ridge]
    gates = {name:max(0.,sorted(values)[math.ceil(error_quantile*len(values))-1]) for name,values in errors.items()}
    bounds = {name:[[min(v[name][j] for v in vectors),max(v[name][j] for v in vectors)] for j in range(len(FEATURES))]
              for name in names if name!='materialize'}
    return dict(schema_version=1,kind='candidate_ridge_log_ratio',metric=metric,
                features=list(FEATURES),coefficients=_fit([p for g in groups for p in g],ridge),ridge=ridge,
                validation_scores={str(r):score(r) for r in RIDGES},validation_residuals=errors,
                error_gates=gates,error_quantile=error_quantile,min_advantage=min_advantage,
                bounds=bounds,bounds_slack=.2,context=context,layouts=layouts,
                compiler_fingerprint=first['compiler_fingerprint'],source_sha256=first['source_sha256'],
                sm_count=first['sm_count'],modules={i['name']:i['module_sha256'] for i in first['candidates']},
                training_bindings=[r['comparison']['bindings'] for r in rows],automatic_dispatch=False,
                limitations=['LOSO residual gate is empirical, not a calibrated confidence interval',
                             'scan/spill interactions are proxies, not profiler attribution',
                             'no automatic lowering policy promotion; materialize remains default'])


def predict_candidate_policy(policy, comparison):
    """Rank candidates without consulting timing samples; abstain conservatively."""
    def finite(v):
        return type(v) in {int,float} and math.isfinite(v)
    if (not isinstance(policy,dict) or policy.get('schema_version')!=1
            or policy.get('kind')!='candidate_ridge_log_ratio' or policy.get('features')!=list(FEATURES)
            or policy.get('metric') not in {'graph_us','wall_us'}
            or not isinstance(policy.get('coefficients'),list) or len(policy['coefficients'])!=len(FEATURES)
            or any(not finite(v) for v in policy['coefficients'])
            or any(k not in policy for k in ['compiler_fingerprint','source_sha256','sm_count','context','layouts','modules'])
            or not isinstance(policy.get('modules'),dict) or 'materialize' not in policy['modules']
            or not isinstance(policy.get('bounds'),dict) or not isinstance(policy.get('error_gates'),dict)
            or not finite(policy.get('bounds_slack')) or policy['bounds_slack']<0
            or not finite(policy.get('min_advantage')) or not 0<=policy['min_advantage']<1):
        raise ValueError('invalid candidate cost policy')
    for name in policy['modules']:
        if name=='materialize':
            continue
        bounds=policy['bounds'].get(name);gate=policy['error_gates'].get(name)
        if (not finite(gate) or gate<0 or not isinstance(bounds,list) or len(bounds)!=len(FEATURES)
                or any(not isinstance(b,list) or len(b)!=2 or not all(finite(v) for v in b) or b[0]>b[1] for b in bounds)):
            raise ValueError('invalid candidate cost bounds/error gate')
    if any(policy[k] != comparison[k] for k in ['compiler_fingerprint','source_sha256','sm_count']):
        raise ValueError('policy source/flowgpu/compiler/SM identity mismatch')
    fallback = dict(candidate='materialize',reason='no eligible candidate has sufficient predicted advantage',ranking=[])
    try:
        if _context(comparison) != policy['context']:
            return dict(fallback,reason='device/dtype context mismatch; materialize fallback')
        if _layout(comparison) not in policy['layouts']:
            return dict(fallback,reason='unseen input layout; materialize fallback')
        control = next(i for i in comparison['candidates'] if i['name']=='materialize')
        if control['module_sha256'] != policy['modules']['materialize']:
            return dict(fallback,reason='materialize module mismatch; materialize fallback')
        base = candidate_cost_features(control,comparison['sm_count'])
    except ValueError as error:
        return dict(fallback,reason=str(error)+'; materialize fallback')
    ranking = []
    for item in comparison['candidates']:
        name = item['name']
        if name=='materialize':
            continue
        entry = dict(candidate=name,eligible=False)
        if policy['modules'].get(name) != item['module_sha256']:
            entry['reason']='candidate module was not fitted'
        else:
            try:
                features = candidate_cost_features(item,comparison['sm_count'])
                x = [features[f]-base[f] for f in FEATURES]
                contributions = {f:v*c for f,v,c in zip(FEATURES,x,policy['coefficients'])}
                prediction = sum(contributions.values()); conservative = prediction+policy['error_gates'][name]
                outside = [f for f,v,(lo,hi) in zip(FEATURES,x,policy['bounds'][name])
                           if v < lo-policy['bounds_slack']*max(hi-lo,1e-6) or v > hi+policy['bounds_slack']*max(hi-lo,1e-6)]
                entry.update(features=features,relative_features=dict(zip(FEATURES,x)),
                             log_ratio_contributions=contributions,predicted_log_ratio=prediction,
                             empirical_error_gate=policy['error_gates'][name],conservative_log_ratio=conservative)
                if outside:
                    entry['reason']='outside screen feature range: '+', '.join(outside)
                elif conservative >= -math.log1p(policy['min_advantage']):
                    entry['reason']='predicted gain insufficient after empirical error gate'
                else:
                    entry.update(eligible=True,reason='inside screen feature range; predicted gain passes empirical error gate')
            except ValueError as error:
                entry['reason']=str(error)
        ranking.append(entry)
    eligible = [r for r in ranking if r['eligible']]
    if not eligible:
        return dict(fallback,ranking=ranking)
    selected = min(eligible,key=lambda r:(r['conservative_log_ratio'],r['candidate']))
    return dict(candidate=selected['candidate'],reason=selected['reason'],ranking=ranking)
