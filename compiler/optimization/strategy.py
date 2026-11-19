"""Explainable reduction-strategy comparisons; never changes compilation defaults.

Logical traffic and program-count bounds are structural features, not latency or
occupancy predictions. Resource/latency observations apply only to the recorded
module, shape, compiler and device context. A separate, opt-in single-split
policy can be fitted on screening measurements and frozen before holdout timing.
"""
import ast
import hashlib
import math
from dataclasses import dataclass

from compiler.shared import product, group_results
from compiler.shared import compiler_fingerprint
from compiler.backend.platform_configs import resolve_platform


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    mode: str = 'materialize'
    columns: int | None = None
    rows: int | None = None
    budget: int | None = None
    warps: int | None = None

    def options(self):
        return dict(cross_domain_strategy=self.mode, platform_config=resolve_platform(
            indexed_columns=self.columns, indexed_rows=self.rows,
            reduction_block_limit=self.budget, num_warps=self.warps))


def candidates(*, recompute=False, large_tiles=False, segmented=False, merge_statistics=False,
               wide_columns=False):
    if type(merge_statistics) is not bool or (merge_statistics and not segmented):
        raise ValueError('merged statistics require explicit segmented opt-in')
    if type(segmented) is not bool:

        raise ValueError('segmented candidate opt-in must be boolean')
    if type(wide_columns) is not bool:
        raise ValueError('wide column candidate opt-in must be boolean')
    if segmented and not recompute:
        raise ValueError('segmented candidates require explicit recompute opt-in')
    if wide_columns and not large_tiles:
        raise ValueError('wide column candidates require explicit large tile opt-in')
    if large_tiles and not recompute:
        raise ValueError('large tile comparison requires explicit recompute opt-in')
    result = [StrategyCandidate('materialize'), StrategyCandidate('materialize-r512-d8', columns=512, rows=8)]
    if recompute:
        result.extend(StrategyCandidate(f'recompute-r{r}-d{d}', 'recompute', r, d)
                      for r, d in [(128, 32), (512, 8), (1024, 4)])
    if large_tiles:
        result.extend(StrategyCandidate(f'recompute-r{r}-d{d}-budget32768', 'recompute', r, d, 32768, 8)
                      for r, d in [(8192, 4), (4096, 8)])
    if wide_columns:
        # The emitted column width is min(dcap, budget // rcap), so a budget of
        # 32768 against a 4096-row tile pins the width to 8, and raising dcap
        # alone does nothing: the budget has to move with it. Width 16 matters
        # because at width 8 a padded row stride leaves four contiguous
        # elements per thread, and a pointer that is not 16-byte aligned drops
        # that to one; width 16 leaves enough contiguous work per thread that
        # the caller's alignment stops deciding the kernel's speed.
        result.extend(StrategyCandidate(f'recompute-r{r}-d{d}-budget{budget}', 'recompute', r, d, budget, 8)
                      for r, d, budget in [(4096, 16, 65536), (2048, 16, 65536)])
    if segmented:
        result.append(StrategyCandidate('segmented-r256-d8','segmented',256,8))
    if merge_statistics:
        result.append(StrategyCandidate('merged-r256-d8','segmented-merge',256,8))
    return tuple(result)


def launch_value(expression, bindings):
    """Evaluate only the integer expression vocabulary emitted in launch grids."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in bindings:
            value = bindings[node.id]
            if type(value) is not int:
                raise ValueError('launch bindings must be integers')
            return value
        if isinstance(node, ast.Tuple):
            return tuple(visit(x) for x in node.elts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            operations = {ast.Add: lambda a,b:a+b, ast.Sub: lambda a,b:a-b,
                          ast.Mult: lambda a,b:a*b, ast.FloorDiv: lambda a,b:a//b}
            if type(node.op) in operations:
                return operations[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.Call) and not node.keywords:
            name = ast.unparse(node.func)
            args = [visit(x) for x in node.args]
            if name in {'min', 'max'} and args:
                return (min if name == 'min' else max)(args)
            if name in {'triton.cdiv', 'tl.cdiv'} and len(args) == 2:
                return -(-args[0] // args[1])
            if name == 'triton.next_power_of_2' and len(args) == 1 and args[0] > 0:
                return 1 << (args[0] - 1).bit_length()
        raise ValueError('unsupported launch expression: ' + ast.unparse(node))
    return visit(ast.parse(expression, mode='eval').body)


def _kernel_features(kernel, bindings, sm_count):
    context = dict(bindings)
    for parameter, arg in zip(kernel.parameters, kernel.launch):
        if arg.kind == 'value':
            context[parameter.split(':')[0].strip()] = launch_value(arg.text, bindings)
    grid = launch_value(kernel.grid, context)
    grid = grid if isinstance(grid, tuple) else (grid,)
    if not grid or any(type(v) is not int or v <= 0 for v in grid):
        raise ValueError('comparison requires a positive static grid')
    programs = math.prod(grid)
    sweeps = []
    for node in ast.walk(ast.parse(kernel.source)):
        if isinstance(node, ast.For):
            if not (isinstance(node.iter, ast.Call) and ast.unparse(node.iter.func) == 'range'
                    and len(node.iter.args) == 1):
                raise ValueError('comparison only supports sequential single-bound sweeps')
            sweeps.append(launch_value(ast.unparse(node.iter.args[0]), context))
    # Reduction emitters place the R/column and output-row/D constexpr bounds
    # last. Read those emitted launch arguments, rather than repeating defaults.
    tile = [launch_value(arg.text, bindings) for arg in kernel.launch[-2:]
            if arg.kind == 'value'] if sweeps else []
    return dict(name=kernel.name, grid=list(grid), programs=programs,
                programs_per_sm=programs / sm_count if sm_count else None,
                active_sm_fraction_upper_bound=min(programs, sm_count) / sm_count if sm_count else None,
                one_program_per_sm_rounds=math.ceil(programs / sm_count) if sm_count else None,
                sequential_sweeps=len(sweeps), chunks_per_sweep=sweeps,
                chunk_iterations_per_program=sum(sweeps), program_chunk_iterations=programs * sum(sweeps),
                tile=tile, tile_elements=math.prod(tile) if tile else None,
                num_warps=dict(kernel.launch_options).get('num_warps'),
                resources=None)


def analyze_candidate(compiled, bindings, *, sm_count=None):
    from compiler.analysis.traffic import plan_traffic
    from compiler.ir.dataflow import INPUT, OUTPUT
    from compiler.backend.triton_backend import module
    if sm_count is not None and (type(sm_count) is not int or sm_count <= 0):
        raise ValueError('sm_count must be a positive integer')
    if any(type(v) is not int or v <= 0 for v in bindings.values()):
        raise ValueError('shape bindings must be positive integers')
    graph = compiled.dataflow
    required = {d for t in graph.values.values() for d in t.shape if not str(d).isdigit()}
    if required != set(bindings):
        raise ValueError(f'comparison requires exactly these symbolic dimensions: {sorted(required)}')
    if not compiled.triton.complete:
        raise ValueError('comparison requires a complete emitted program')
    code = module([compiled.triton])
    kernels = [_kernel_features(k, bindings, sm_count) for k in compiled.triton.kernels]
    stages=compiled.memory.stage_plan
    for kernel, feature in zip(compiled.triton.kernels, kernels):
        if stages is not None:
            index=next(i for i,s in enumerate(stages.stages) if kernel.name.endswith('_'+s.kind))
            stage=stages.stages[index]
            r=launch_value(stages.r_tile,bindings);d=launch_value(stages.d_tile,bindings)
            parts=launch_value(stages.segments,bindings)
            if stages.merge is None:
                tile=[r] if index==0 else [r,d] if index==1 else [1<<(parts-1).bit_length(),d]
                base_scans=[1,launch_value(stages.d_tiles,bindings),0][index]
                wide_scans=[0,1,0][index]
            else:
                tile=[r,d] if index==0 else [1<<(parts-1).bit_length(),d]
                base_scans=2*launch_value(stages.d_tiles,bindings) if index==0 else 0
                wide_scans=1 if index==0 else 0
                feature.update(conditional_original_base_full_r_scans=launch_value(stages.d_tiles,bindings) if index==1 else 0,
                               conditional_original_wide_full_r_scans=1 if index==1 else 0,
                               exceptional_replay=index==1)
            feature.update(stage=stage.kind,stage_inputs=list(stage.inputs),stage_outputs=list(stage.outputs),
                           tile=tile,tile_elements=math.prod(tile),segments=parts,
                           original_base_full_r_scans=base_scans,original_wide_full_r_scans=wide_scans)
            continue
        group = next(g for g in compiled.fusion.groups if kernel.results == group_results(graph, g))
        reductions = [n for n in group.members if n.op in {'sum_last', 'max_last'}]
        if all(n.op == 'gather_rows' for n in group.members):
            from compiler.analysis.precision import DTYPE_BYTES
            from compiler.optimization.row_gather_fusion import group_config
            config = group_config(group, compiled.triton.platform_config)
            rows = launch_value(product(graph.values[group.members[0].inputs[1]].shape), bindings)
            widths = [launch_value(product(graph.values[n.inputs[0]].shape[1:]), bindings)
                      for n in group.members]
            if config.gather_mapping == 'row_tiled':
                tile = [config.indexed_rows or 8, config.indexed_columns or 128]
                index_reads = rows * feature['grid'][1]
            else:
                tile = [config.default_block_size]
                index_reads = rows * max(widths)
            logical_elements = rows * sum(widths)
            feature.update(gather_mapping=config.gather_mapping, output_rows=rows,
                           gather_outputs=[n.output for n in group.members],
                           output_widths=widths, tile=tile, tile_elements=math.prod(tile),
                           useful_lane_fraction=logical_elements / (feature['programs'] * math.prod(tile) * len(widths)),
                           logical_index_loads=index_reads,
                           logical_index_bytes=index_reads * DTYPE_BYTES[graph.values[group.members[0].inputs[1]].dtype],
                           source_read_bytes_if_valid=sum(rows*w*DTYPE_BYTES[graph.values[n.inputs[0]].dtype]
                                                         for n,w in zip(group.members,widths)),
                           note='index load instructions before cache effects; source byte upper bound assumes valid rows; no unique-index or HBM claim')
        if reductions and len(feature['tile']) == 2:
            domain = graph.values[reductions[-1].inputs[0]].shape
            width = launch_value(str(domain[-1]), bindings)
            count = launch_value(product(domain[:-1]), bindings)
            r, d = feature['tile']
            feature.update(domain=list(domain), logical_width=width, logical_output_rows=count,
                           useful_lane_fraction=count*width/(feature['programs']*r*d*math.ceil(width/r)))
    traffic = plan_traffic(graph, compiled.fusion, compiled.memory, tiled=True)
    outputs = {n.inputs[0] for n in graph.nodes if n.op == OUTPUT}
    intermediates = [{'value':b.value, 'bytes':launch_value(b.size, bindings)} for b in compiled.memory.buffers
                     if b.storage is not None and b.storage != 'input' and b.value not in outputs]
    slots = [max(launch_value(b.size, bindings) for b in compiled.memory.buffers if b.storage == f'slot{s.index}')
             for s in compiled.memory.slots]
    scratch=[{'value':s.name,'bytes':launch_value(s.size,bindings),'lifetime_stages':[s.writer,max(s.readers)]}
             for s in stages.scratch] if stages is not None else []
    intermediates.extend(scratch)
    domains = [list(graph.values[n.inputs[0]].shape) for g in compiled.fusion.groups for n in g.members
               if n.op in {'sum_last', 'max_last'}]
    return dict(module_sha256=hashlib.sha256(code.encode()).hexdigest(),
                launches=len(kernels), slots=len(slots), allocated_slot_bytes=sum(slots),
                scratch_allocations=len(scratch),allocated_bytes=sum(slots)+sum(s['bytes'] for s in scratch),
                intermediate_values=intermediates, intermediate_bytes=sum(v['bytes'] for v in intermediates),
                reduction_domains=domains, cross_domain_applied=any(g.cross_domain for g in compiled.fusion.groups),
                segmented_applied=stages is not None,merged_statistics_applied=stages is not None and stages.merge is not None,
                logical_traffic_bytes=launch_value(traffic.moved_expression, bindings),
                normal_path_logical_traffic_bytes=sum(c.count*launch_value(c.size,bindings) for c in traffic.charges if not c.reason.startswith('conditional replay')),
                conditional_replay_logical_traffic_bytes=sum(c.count*launch_value(c.size,bindings) for c in traffic.charges if c.reason.startswith('conditional replay')),
                interface_baseline_bytes=launch_value(traffic.floor_expression, bindings),
                traffic_charges=[dict(value=c.value, bytes=c.count*launch_value(c.size, bindings), reason=c.reason)
                                 for c in traffic.charges], kernels=kernels,
                inputs={n.output:dict(dtype=graph.values[n.output].dtype,
                            shape=[launch_value(str(d), bindings) for d in graph.values[n.output].shape])
                        for n in graph.nodes if n.op == INPUT},
                observation=None, evidence_status='structural', evidence_rejections=[])


def _attach_observation(item, name, bindings, evidence, source_hash, fingerprint, sm_count):
    if evidence is None:
        return
    if not isinstance(evidence, dict) or not isinstance(evidence.get('records', []), list):
        raise ValueError('evidence must be an object with a records list')
    if not isinstance(evidence.get('device', {}), dict) or any(not isinstance(r, dict) for r in evidence.get('records', [])):
        raise ValueError('evidence device and records must be objects')
    required = [('schema_version', 1), ('source_sha256', source_hash), ('compiler_fingerprint', fingerprint)]
    for key, expected in required:
        if evidence.get(key) != expected:
            item['evidence_rejections'].append('mismatched ' + key)
    device = evidence.get('device', {})
    if not device.get('name') or not device.get('uuid') or device.get('sm_count') != sm_count:
        item['evidence_rejections'].append('device identity/SM context missing or mismatched')
    if item['evidence_rejections']:
        return
    records = [r for r in evidence.get('records', []) if r.get('candidate') == name and r.get('bindings') == bindings]
    if len(records) != 1:
        item['evidence_rejections'].append('no unique observation for this candidate and shape')
        return
    record = records[0]
    if record.get('module_sha256') != item['module_sha256']:
        item['evidence_rejections'].append('mismatched generated module')
        return
    context = record.get('input_context')
    if not isinstance(context, dict) or not context or record.get('correctness_passed') is not True:
        item['evidence_rejections'].append('input context or correctness evidence missing')
        return
    if set(context) != set(item['inputs']) or any(
        not isinstance(context[name], dict) or context[name].get('dtype') != expected['dtype'] or context[name].get('shape') != expected['shape']
        or not isinstance(context[name].get('stride'), list)
        or len(context[name]['stride']) != len(expected['shape'])
        or any(type(s) is not int or s < 0 for s in context[name]['stride'])
        for name, expected in item['inputs'].items()
    ):
        item['evidence_rejections'].append('input dtype/shape/stride context mismatch')
        return
    resources = record.get('kernels', [])
    if not isinstance(resources, list) or any(not isinstance(r, dict) for r in resources):
        item['evidence_rejections'].append('invalid kernel observations')
        return
    if [r.get('name') for r in resources] != [k['name'] for k in item['kernels']]:
        item['evidence_rejections'].append('kernel observations do not match emitted launches')
        return
    if any(resource.get(key) is not None and (type(resource[key]) is not int or resource[key] < 0)
           for resource in resources for key in ['n_regs', 'n_spills_reported', 'shared_reported']):
        item['evidence_rejections'].append('invalid resource counters')
        return
    has_timings = bool(record.get('samples'))
    if has_timings and not isinstance(record['samples'], dict):
        item['evidence_rejections'].append('invalid timing observations')
        return
    for metric in ['graph_us', 'wall_us'] if has_timings else []:
        samples = record.get('samples', {}).get(metric, [])
        if not isinstance(samples, list) or not samples or any(type(s) not in {int, float} or not math.isfinite(s) or s <= 0 for s in samples):
            item['evidence_rejections'].append('invalid or missing ' + metric + ' samples')
    if item['evidence_rejections']:
        return
    for kernel, resource in zip(item['kernels'], resources):
        kernel['resources'] = {key:resource.get(key) for key in ['n_regs', 'n_spills_reported', 'shared_reported', 'num_warps']}
    item['observation'] = dict(device=device, input_context=context, samples=record.get('samples', {}),
                               split=record.get('split'), note='only for this recorded execution context')
    item['evidence_status'] = 'measured_for_recorded_context' if has_timings else 'compiled_resources_for_recorded_context'


def compare_strategies(source, bindings, *, recompute=False, large_tiles=False, segmented=False, merge_statistics=False, sm_count=None,
                       evidence=None, compile_options=None):
    return _compare_strategies(source, bindings, recompute=recompute, large_tiles=large_tiles,
        segmented=segmented,merge_statistics=merge_statistics,sm_count=sm_count, evidence=evidence, compile_options=compile_options)


def _compare_strategies(source, bindings, *, recompute=False, large_tiles=False, segmented=False, merge_statistics=False, sm_count=None,
                        evidence=None, compile_options=None, inspect_candidate=None):
    """One compile per candidate; private visitor can verify the same program.

    The visitor is scoped to this call. No cross-call compiler-object cache or
    trusted evidence shortcut is introduced.
    """
    from compiler.compile import compile_source
    options = dict(compile_options or {})
    if any(k in options for k in ['shape_bindings', 'cross_domain_strategy', 'platform_config']):
        raise ValueError('candidate configurations and modeling shapes are owned by the comparison')
    if options.get('gpu_backend', 'triton') != 'triton' or not options.get('index_fusion', True) or not options.get('reduction_fusion', True):
        raise ValueError('strategy comparison requires native CUDA indexed/reduction fusion')
    fingerprint = compiler_fingerprint()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    items = []
    for candidate in candidates(recompute=recompute, large_tiles=large_tiles, segmented=segmented,merge_statistics=merge_statistics):
        compiled = compile_source(source, **options, **candidate.options())
        if len(compiled.pipelines) != 1:
            raise ValueError('strategy comparison requires one pipeline')
        item = analyze_candidate(compiled.pipelines[0], bindings, sm_count=sm_count)
        item.update(name=candidate.name, mode=candidate.mode,
                    config=compiled.pipelines[0].triton.platform_config.as_dict())
        if inspect_candidate is not None:
            inspect_candidate(candidate, compiled.pipelines[0])
        _attach_observation(item, candidate.name, bindings, evidence, source_hash, fingerprint, sm_count)
        items.append(item)
    # Analytic dominance has a narrow meaning: lower in every named structural
    # component, not a prediction that one candidate will execute faster.
    def vector(item):
        return (item['launches'], item['logical_traffic_bytes'],
                sum(k['program_chunk_iterations'] for k in item['kernels']),
                max((k['tile_elements'] or 0 for k in item['kernels']), default=0))
    frontier = [i['name'] for i in items if not any(
        all(a <= b for a,b in zip(vector(j), vector(i))) and vector(j) != vector(i)
        for j in items if j is not i)]
    return dict(schema_version=1, compiler_fingerprint=fingerprint, source_sha256=source_hash,
                bindings=dict(bindings), sm_count=sm_count, default='materialize', automatic_dispatch=False,
                candidates=items, structural_frontier=frontier,
                frontier_components=['launches', 'logical_traffic_bytes', 'program_chunk_iterations', 'max_tile_elements'],
                limitations=['program count gives an active-SM upper bound, not occupancy or a measured wave count',
                             'tile_elements is not register count; resource observations are exact-context only',
                             'logical traffic and the structural frontier do not predict latency',
                             'comparison shape bindings do not specialize the generated modules'])


# Fixed before holdout measurements: one split, at least three distinct screen
# shapes in each leaf. Neither R alone nor holdout latency can define a branch.
POLICY_FEATURES = ('programs_per_sm', 'chunk_iterations_per_program', 'logical_traffic_ratio',
                   'tile_elements', 'n_regs', 'n_spills_reported', 'shared_reported')


def policy_features(comparison, probe='recompute-r512-d8'):
    item = next(i for i in comparison['candidates'] if i['name'] == probe)
    control = next(i for i in comparison['candidates'] if i['name'] == 'materialize')
    kernels = item['kernels']
    result = {key:max((k[key] for k in kernels if k[key] is not None), default=None)
              for key in ['programs_per_sm', 'chunk_iterations_per_program', 'tile_elements']}
    result['logical_traffic_ratio'] = item['logical_traffic_bytes'] / control['logical_traffic_bytes']
    for key in ['n_regs', 'n_spills_reported', 'shared_reported']:
        values = [(k.get('resources') or {}).get(key) for k in kernels]
        result[key] = max(values) if values and all(type(v) is int and v >= 0 for v in values) else None
    return result


def fit_screen_policy(rows, metric, *, probe='recompute-r512-d8', min_leaf=3, min_gain=.05):
    """Fit a small measured policy; rows must be screening observations only.

    Loss is weighted mean log latency ratio to materialize, separately for each
    metric. An optional split needs >=5% training gain over a single candidate.
    These are explicit regularization choices, not hardware performance laws.
    """
    import statistics
    if metric not in {'graph_us', 'wall_us'}:
        raise ValueError('unknown timing objective')
    if len({tuple(sorted(r['comparison']['bindings'].items())) for r in rows}) != len(rows):
        raise ValueError('screen rows must have distinct shapes')
    names = [i['name'] for i in rows[0]['comparison']['candidates']] if rows else []
    if not names:
        raise ValueError('screen rows required')
    for row in rows:
        if row.get('split') != 'screen' or [i['name'] for i in row['comparison']['candidates']] != names:
            raise ValueError('policy fitting requires identical candidates and screening rows only')
        if (row['comparison']['compiler_fingerprint'] != rows[0]['comparison']['compiler_fingerprint'] or
                row['comparison']['source_sha256'] != rows[0]['comparison']['source_sha256']):
            raise ValueError('screen rows must share source/compiler identity')
        if type(row.get('weight', 1)) not in {int,float} or not math.isfinite(row.get('weight', 1)) or row.get('weight', 1) <= 0:
            raise ValueError('screen weights must be positive and finite')
        for name in names:
            v = row['medians'][name][metric]
            if not math.isfinite(v) or v <= 0:
                raise ValueError('latencies must be positive and finite')
    def loss(subset, name):
        return sum(r.get('weight',1)*math.log(r['medians'][name][metric]/r['medians']['materialize'][metric]) for r in subset)
    def best(subset):
        return min(names, key=lambda name:(loss(subset,name), names.index(name)))
    fixed = best(rows); weight = sum(r.get('weight',1) for r in rows)
    fixed_loss = loss(rows, fixed); winner = None; winner_loss = fixed_loss
    features = [policy_features(r['comparison'], probe) for r in rows]
    for feature in POLICY_FEATURES:
        values = [f[feature] for f in features]
        if any(v is None for v in values):
            continue
        distinct = sorted(set(values))
        for a,b in zip(distinct, distinct[1:]):
            threshold = (a+b)/2
            left = [r for r,v in zip(rows,values) if v <= threshold]
            right = [r for r,v in zip(rows,values) if v > threshold]
            if min(len(left),len(right)) < min_leaf:
                continue
            ln,rn = best(left),best(right); cost = loss(left,ln)+loss(right,rn)
            if cost < winner_loss:
                winner_loss=cost; winner=dict(feature=feature, threshold=threshold, left=ln, right=rn,
                                             left_shapes=len(left), right_shapes=len(right))
    gain = math.exp((fixed_loss-winner_loss)/weight)
    if gain < 1+min_gain:
        winner=None
    return dict(schema_version=1, metric=metric, probe=probe, fixed_candidate=fixed,
                split=winner, training_gain_vs_fixed=gain, min_leaf=min_leaf, min_gain=min_gain,
                training_bindings=[r['comparison']['bindings'] for r in rows],
                compiler_fingerprint=rows[0]['comparison']['compiler_fingerprint'],
                source_sha256=rows[0]['comparison']['source_sha256'], features=list(POLICY_FEATURES),
                automatic_dispatch=False)


def predict_screen_policy(policy, comparison):
    if (policy['compiler_fingerprint'] != comparison['compiler_fingerprint'] or
            policy['source_sha256'] != comparison['source_sha256']):
        raise ValueError('policy source/compiler identity mismatch')
    split = policy['split']
    if split is None:
        return dict(candidate=policy['fixed_candidate'], reason='fixed screen-only distribution winner')
    value = policy_features(comparison, policy['probe'])[split['feature']]
    if value is None:
        return dict(candidate='materialize', reason='required measured feature unavailable; conservative fallback')
    branch = 'left' if value <= split['threshold'] else 'right'
    return dict(candidate=split[branch], reason=f"{split['feature']}={value} {'<=' if branch=='left' else '>'} {split['threshold']}")
