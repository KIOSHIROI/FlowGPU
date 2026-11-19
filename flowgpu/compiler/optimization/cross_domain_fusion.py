"""Opt-in recomputation across one inserted broadcast axis.

Recognizes domains and dependency boundaries, not operator/pipeline names.
Only a closed pair of regions can merge; extra readers/exports retain the
materialized partition. No automatic latency decision is made here.
"""
from flowgpu.compiler.optimization.fusion import (FusionGroup, FusionReport,
                                           order_by_dependencies)
from flowgpu.compiler.shared import group_results
from flowgpu.compiler.optimization.indexed_fusion import REDUCTIONS, supported
from flowgpu.compiler.optimization.reduction_fusion import row_key, full_key


def fuse_cross_domain(graph, report, config):
    groups=list(report.groups);index=graph.index_of
    for base in list(groups):
        if base not in groups or not base.reduction_domain:continue
        base_domain=tuple(base.reduction_domain)
        base_row=row_key(base_domain[:-1]);width=str(base_domain[-1])
        exports=group_results(graph,base)
        for wide in list(groups):
            if wide is base or wide not in groups:continue
            reductions=[n for n in wide.members if n.op in REDUCTIONS]
            if len(reductions)!=1 or reductions[0].op!='sum_last':continue
            domain=tuple(graph.values[reductions[0].inputs[0]].shape)
            # Keep explicit prefix axes: a permuted or collapsed row domain
            # cannot be substituted just because the element counts agree.
            if len(domain)!=len(base_domain)+1 or tuple(map(str,domain[:-2]))!=tuple(map(str,base_domain[:-1])) or str(domain[-1])!=width:continue
            members=tuple(sorted(base.members+wide.members,key=index));inside={index(n) for n in members}
            if any(not supported(n.op) or n.op in {'gather_row_offset','gather_rows'} for n in members):continue
            if any(index(c) not in inside and c.op!='output' for v in exports for c in graph.consumers_of(v)):continue
            # Full-domain exports need a global tensor; this experiment only
            # supports row outputs, preserving the established fallback.
            if any(full_key(graph.values[v].shape)==full_key(base_domain)
                   and row_key(graph.values[v].shape)!=base_row
                   and any(c.op=='output' for c in graph.consumers_of(v)) for v in exports):continue
            if not any(v in exports for n in wide.members for v in n.inputs):continue
            start=min(inside)
            if any(index(p)>=start for n in members for v in n.inputs
                   if (p:=graph.producer_of(v)) is not None and index(p) not in inside):continue
            # Base statistics enter the wide domain through an aligned
            # trailing singleton/reshape. Computed row permutations stay out.
            unsafe=False
            def input_view(v):
                p=graph.producer_of(v)
                if p is None or p.op=='input':return True
                return p.op in {'to_fp32','cast_like','cast_to'} and input_view(p.inputs[0])
            for n in wide.members:
                if n.op=='broadcast_like':
                    source=graph.values[n.inputs[0]].shape;target=graph.values[n.output].shape
                    padded=['1']*(len(target)-len(source))+list(map(str,source))
                    if full_key(source)==full_key(base_domain) and full_key(target)==full_key(domain):
                        if tuple(padded[:-2])!=tuple(map(str,domain[:-2])) or padded[-2]!='1' or padded[-1]!=width:unsafe=True
                if n.op=='move_first_last' and not input_view(n.inputs[0]):unsafe=True
            if unsafe:continue
            fused=tuple(n.output for n in members if all(index(c) in inside for c in graph.consumers_of(n.output)))
            anchor=reductions[0]
            base_reductions=[n for n in members if n.op in REDUCTIONS and n is not anchor]
            batches=tuple((n.output,) for n in base_reductions+[anchor])
            # Different reduction domains can share a sweep when the final
            # scalar sum is not an input to the wide sum. Keep the dependency
            # decision in the IR, consumed by emission and traffic accounting.
            producers={n.output:n for n in members}
            dependency_cache={}
            def depends(value,needle):
                if value==needle:return True
                if (value,needle) in dependency_cache:return dependency_cache[value,needle]
                node=producers.get(value)
                if node is None or node.op in {'ones_like','zeros_like','neg_inf_like'}:return False
                inputs=node.inputs[:1] if node.op in {'reshape','cast_like','cast_to','to_fp32','broadcast_like','move_first_last'} else node.inputs
                result=any(depends(v,needle) for v in inputs)
                dependency_cache[value,needle]=result
                return result
            if (config.coalesce_reduction_sweeps and base_reductions
                    and base_reductions[-1].op=='sum_last'
                    and not depends(anchor.inputs[0],base_reductions[-1].output)):
                batches=batches[:-2]+((base_reductions[-1].output,anchor.output),)
            limit=config.reduction_block_limit or 4096
            rcap=min(limit,config.indexed_columns or 128)
            dcap=min(config.indexed_rows or 32,max(1,limit//rcap))
            merged=FusionGroup(anchor,members,fused,
                tuple(n for n in members if index(n)<index(anchor)),
                tuple(n for n in members if index(n)>index(anchor)),
                indexed=True,reduction_domain=domain,reduction_batches=batches,
                cross_domain=base_domain,cross_tile=(rcap,dcap))
            result=group_results(graph,merged)
            wide_row=row_key(domain[:-1])
            if any(row_key(graph.values[v].shape) not in (base_row,wide_row) for v in result):continue
            groups=[g for g in groups if g is not base and g is not wide]+[merged]
            groups=list(order_by_dependencies(graph, groups, index_of=index));break
    return FusionReport(tuple(groups),frozenset(v for g in groups for v in g.fused_values))
