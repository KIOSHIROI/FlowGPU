"""Bound, explainable gather launch comparisons. No automatic dispatch.

Each candidate changes compiler options, not generated source. Observations use
the same source/compiler/module/device/layout checks as reduction comparisons.
"""
import hashlib
from compiler.shared import compiler_fingerprint
from compiler.compile import compile_source
from compiler.backend.platform_configs import resolve_platform
from compiler.optimization.strategy import analyze_candidate, _attach_observation


def candidates(source=None, bindings=None, *, partial=False):
    result = {
        'flat-separate': dict(gather_fusion=False, platform_config=resolve_platform()),
        'flat-fused': dict(gather_fusion=True, platform_config=resolve_platform()),
        'tiled-separate-8x64': dict(gather_fusion=False, platform_config=resolve_platform(
            gather_mapping='row_tiled', indexed_rows=8, indexed_columns=64)),
        'tiled-fused-8x64': dict(gather_fusion=True, platform_config=resolve_platform(
            gather_mapping='row_tiled', indexed_rows=8, indexed_columns=64)),
        'tiled-fused-4x256': dict(gather_fusion=True, platform_config=resolve_platform(
            gather_mapping='row_tiled', indexed_rows=4, indexed_columns=256)),
    }

    if partial:
        if source is None or bindings is None:
            raise ValueError('partial gather candidates require source and modeling bindings')
        from compiler.shared import product
        from compiler.optimization.strategy import launch_value
        graph = compile_source(source).pipelines[0].dataflow
        buckets = {}
        for node in graph.nodes:
            if node.op == 'gather_rows':
                buckets.setdefault(node.inputs[1], []).append(node)
        partitions = []
        for nodes in buckets.values():
            if len(nodes) < 3:
                continue
            start = min(graph.index_of(n) for n in nodes)
            nodes = [n for n in nodes if all(graph.index_of(graph.producer_of(v)) < start for v in n.inputs)]
            if len(nodes) < 3:
                continue
            widest = max(nodes, key=lambda n: launch_value(product(graph.values[n.inputs[0]].shape[1:]), bindings))
            partitions.append(([n.output for n in nodes if n is not widest], [widest.output]))
        for label, narrow, wide in [
            ('partial-flat-flat', ('flat',None,None), ('flat',None,None)),
            ('partial-tiled8x64-flat', ('row_tiled',8,64), ('flat',None,None)),
            ('partial-flat-tiled4x256', ('flat',None,None), ('row_tiled',4,256)),
            ('partial-tiled8x64-tiled4x256', ('row_tiled',8,64), ('row_tiled',4,256)),
        ]:
            plan = [dict(outputs=outputs, mapping=schedule[0], rows=schedule[1], columns=schedule[2])
                    for small, large in partitions for outputs, schedule in [(small,narrow),(large,wide)]]
            if plan:
                result[label] = dict(gather_plan=plan, platform_config=resolve_platform())
    return result


def gather_features(item):
    """Explainable proxies, never interpreted as cycles or physical traffic."""
    kernels = item['kernels']
    return dict(launches=item['launches'], programs=sum(k['programs'] for k in kernels),
                logical_index_loads=sum(k.get('logical_index_loads',0) for k in kernels),
                source_bytes_if_valid=sum(k.get('source_read_bytes_if_valid',0) for k in kernels),
                lane_waste=sum(k['programs']*k['tile_elements']*len(k.get('output_widths',[]))
                               *(1-k.get('useful_lane_fraction',1)) for k in kernels),
                low_grid_launches=sum(k.get('active_sm_fraction_upper_bound',1)<1 for k in kernels),
                max_registers=max((k['resources']['n_regs'] for k in kernels if k['resources'] and k['resources']['n_regs'] is not None),default=None),
                max_spill=max((k['resources']['n_spills_reported'] for k in kernels if k['resources'] and k['resources']['n_spills_reported'] is not None),default=None))


def compare_gather_strategies(source, bindings, *, sm_count=None, evidence=None, partial=False):
    fingerprint = compiler_fingerprint()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    items = []
    for name, options in candidates(source, bindings, partial=partial).items():
        compiled = compile_source(source, **options)
        if len(compiled.pipelines) != 1:
            raise ValueError('gather comparison requires one pipeline')
        p = compiled.pipelines[0]
        operators = [n.op for n in p.dataflow.nodes if n.op not in {'input', 'output'}]
        if not operators or any(op != 'gather_rows' for op in operators):
            raise ValueError('gather launch comparison currently requires only gather_rows operations')
        item = analyze_candidate(p, bindings, sm_count=sm_count)
        item.update(name=name, config=p.triton.platform_config.as_dict(), gather_fusion=options.get('gather_fusion',False), gather_plan=options.get('gather_plan'))
        _attach_observation(item, name, bindings, evidence, source_hash, fingerprint, sm_count)
        item['decision_features'] = gather_features(item)
        items.append(item)
    return dict(schema_version=1, compiler_fingerprint=fingerprint, source_sha256=source_hash,
                bindings=dict(bindings), sm_count=sm_count, default='flat-separate',
                automatic_dispatch=False, candidates=items,
                limitations=['shared-index fusion reduces launches, not output allocation',
                             'unequal widths may waste lanes and duplicate index loads across column tiles',
                             'logical load counts are not HBM traffic or occupancy',
                             'resource and timing evidence applies only to exact recorded context',
                             'no platform defaults are promoted by comparison'])
