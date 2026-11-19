"""Fuse compositional index expressions without cloning graph operations.

One region has one externally visible result and at most one last-axis
reduction. Shared values stay materialized unless all their readers join the
same region. Index remaps are evaluated at use sites, not expanded in memory.
Legacy GEMM/rowwise placements remain untouched.
"""
from flowgpu.compiler.optimization.fusion import (FusionGroup, FusionReport,
                                           order_by_dependencies, is_pointwise)
from flowgpu.compiler.shared import INDEX_OPS, PRIMITIVE_OPS, VALUE_LAYER_OPS

PRIMITIVES = PRIMITIVE_OPS + VALUE_LAYER_OPS

from flowgpu.compiler.shared import REDUCTIONS  # noqa: E402


def supported(op):
    return op in INDEX_OPS or is_pointwise(op)


def fuse_indexed(graph, report, *, split_views=False, layout_views=False, layout_view_cost=False):
    from flowgpu.compiler.shared import group_results
    groups = list(report.groups)
    index = graph.index_of

    # Opt-in only: a last-axis `split` becomes a remapped index over its source
    # instead of a layout anchor. The two parts are read from one pointer with
    # a constexpr part offset, so the whole consumer region is one kernel. Off
    # by default: the layout-anchor groups stay and every default emission is
    # byte-identical to before.
    extra_ops = {'split', 'concat', 'interleave', 'transpose'} if (split_views or layout_views) else set()

    def supported_here(op):
        return op in extra_ops or supported(op)

    def eligible(group):
        return (not group.row_gather_schedule
                and all(supported_here(n.op) for n in group.members))

    def shape_key(value):
        return tuple(d for d in graph.values[value].shape if str(d) != '1')

    # Reverse sinks first so their ancestors can become prologues; merged
    # groups are never cloned or assigned to a second sink.
    for root in reversed(list(groups)):
        if root not in groups or not eligible(root):
            continue
        outputs = group_results(graph, root)
        if len(outputs) != 1:
            continue
        owner = {n.output: i for i, g in enumerate(groups) for n in g.members}
        root_id = groups.index(root)
        chosen = {root_id}
        reductions = [n for n in root.members if n.op in REDUCTIONS]
        if len(reductions) > 1:
            continue

        def visit(group_id):
            for n in reversed(groups[group_id].members):
                for value in n.inputs:
                    parent = owner.get(value)
                    if parent is None or parent in chosen or not eligible(groups[parent]):
                        continue
                    new_reductions = [x for x in groups[parent].members if x.op in REDUCTIONS]
                    if len(reductions) + len(new_reductions) > 1:
                        continue
                    # Expanding a reduced row back into many elements would
                    # repeat the entire reduction. Leave that boundary intact.
                    if new_reductions and shape_key(new_reductions[0].output) != shape_key(outputs[0]):
                        continue
                    chosen.add(parent)
                    reductions.extend(new_reductions)
                    visit(parent)
        visit(root_id)

        # A diamond can be internal; a reader outside this region cannot.
        # After pruning shared producers, drop ancestors no longer reachable
        # from the sink (their work belongs to the removed producer).
        while True:
            before = set(chosen)
            inside = {index(n) for i in chosen for n in groups[i].members}
            for i in list(chosen - {root_id}):
                if any(index(c) not in inside for n in groups[i].members
                       for c in graph.consumers_of(n.output)):
                    chosen.remove(i)
            reachable = {root_id}
            pending = [root_id]
            while pending:
                for n in groups[pending.pop()].members:
                    for value in n.inputs:
                        parent = owner.get(value)
                        if parent in chosen and parent not in reachable:
                            reachable.add(parent)
                            pending.append(parent)
            chosen &= reachable
            if chosen == before:
                break
        members = sorted((n for i in chosen for n in groups[i].members), key=index)
        has_primitive = any(n.op in PRIMITIVES for n in members)
        has_layout = any(n.op in {'split', 'concat', 'interleave', 'transpose',
                                  'reshape', 'cast_like', 'cast_to', 'to_fp32',
                                  'move_first_last'} for n in members)
        if not has_primitive:
            # P0: a layout anchor plus its pointwise consumer can be one
            # indexed region, provided the caller opted in.  Without this,
            # split+add stays three kernels (two layout copies + pointwise).
            if not layout_views or not has_layout:
                continue
            if layout_view_cost:
                from flowgpu.compiler.backend.backends import product
                from flowgpu.compiler.optimization.layout_view_cost import should_fuse_layout_view
                result_shape = graph.values[outputs[0]].shape
                if not all(str(d).isdigit() for d in result_shape):
                    continue
                numel = int(product(result_shape))
                layout_count = sum(1 for n in members if n.op in {
                    'split', 'concat', 'interleave', 'transpose', 'reshape',
                    'cast_like', 'cast_to', 'to_fp32', 'move_first_last'})
                consumer_count = len(members) - layout_count
                feasible, _costs = should_fuse_layout_view(numel, layout_count,
                                                           max(consumer_count, 1))
                if not feasible:
                    continue
        reductions = [n for n in members if n.op in REDUCTIONS]
        if len(reductions) > 1 or (reductions and shape_key(reductions[0].output) != shape_key(outputs[0])):
            continue
        inside = {index(n) for n in members}
        # Preserve the existing source-order launch schedule and conservative
        # live intervals. A late external operand must not be read early.
        start = min(inside)
        if any(index(p) >= start for n in members for v in n.inputs
               if (p := graph.producer_of(v)) is not None and index(p) not in inside):
            continue
        if len(chosen) == 1 and not root.indexed:
            continue
        if reductions:
            anchor = reductions[0]
        else:
            primitive = next((n for n in reversed(members) if n.op in PRIMITIVES), None)
            if primitive is not None:
                anchor = primitive
            else:
                # layout-view region with no primitive anchor: the last
                # pointwise computation is the sink and names the kernel.
                anchor = next((n for n in reversed(members)
                               if n.op not in {'split', 'concat', 'reshape', 'cast_like',
                                               'cast_to', 'to_fp32', 'move_first_last'}), None)
                if anchor is None:
                    continue
        fused = tuple(n.output for n in members if n.output != outputs[0])
        merged = FusionGroup(anchor, tuple(members), fused,
                             tuple(n for n in members if index(n) < index(anchor)),
                             tuple(n for n in members if index(n) > index(anchor)), indexed=True)
        groups = [g for i, g in enumerate(groups) if i not in chosen] + [merged]
        groups = list(order_by_dependencies(graph, groups, index_of=index))
    return FusionReport(tuple(groups), frozenset(v for g in groups for v in g.fused_values))
