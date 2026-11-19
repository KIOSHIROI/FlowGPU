"""Explicit sibling row-gather fusion; no cloning and no reordered dependencies.

Only bare gathers whose sources/index exist before the first gather may join.
The inputs need not have equal row widths, row counts, ranks or storage dtypes.
This removes launches, not intermediates, and is an opt-in CUDA experiment.
"""
from compiler.optimization.fusion import (FusionGroup, FusionReport,
                                           order_by_dependencies)


def group_config(group, config):
    """One mapping resolver shared by emission and structural comparison."""
    if not group.row_gather_schedule:
        return config
    from dataclasses import replace
    mapping, rows, columns = group.row_gather_schedule
    return replace(config, gather_mapping=mapping, indexed_rows=rows, indexed_columns=columns)


def apply_gather_plan(graph, report, entries, config):
    """Explicit partitions by SSA outputs; dependencies remain source ordered.

    Unmentioned gathers retain their existing mapping. Planned groups are opaque
    to later indexed fusion so the requested launch boundary cannot disappear.
    """
    if not isinstance(entries, (tuple, list)) or not entries:
        raise ValueError('gather_plan requires a nonempty list of group specifications')
    groups = list(report.groups)
    available = {g.anchor.output: g for g in groups if len(g.members) == 1
                 and g.anchor is not None and g.anchor.op == 'gather_rows'}
    used = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {'outputs', 'mapping', 'rows', 'columns'}:
            raise ValueError('gather plan entries require outputs, mapping, rows, columns')
        outputs = entry['outputs']
        if (not isinstance(outputs, (list, tuple)) or not outputs
                or any(not isinstance(v, str) or v not in available for v in outputs)
                or len(set(outputs)) != len(outputs) or used.intersection(outputs)):
            raise ValueError('gather plan outputs must name distinct bare gather SSA values')
        schedule = (entry['mapping'], entry['rows'], entry['columns'])
        if schedule[0] == 'flat' and any(v is not None for v in schedule[1:]):
            raise ValueError('flat gather plan cannot contain row/column tiles')
        if schedule[0] == 'row_tiled' and any(v is None for v in schedule[1:]):
            raise ValueError('row_tiled gather plan requires both tile dimensions')
        selected = [available[v] for v in outputs]
        members = tuple(sorted((g.anchor for g in selected), key=graph.index_of))
        if len({n.inputs[1] for n in members}) != 1:
            raise ValueError('gather partition requires one shared index SSA value')
        start = graph.index_of(members[0])
        if any(graph.index_of(graph.producer_of(v)) >= start for n in members for v in n.inputs):
            raise ValueError('gather partition would read a late dependency')
        joined = FusionGroup(members[0], members, (), row_gather=True, row_gather_schedule=schedule)
        group_config(joined, config)  # validate tiles before changing the partition
        groups = [g for g in groups if g not in selected] + [joined]
        used.update(outputs)
    groups = list(order_by_dependencies(graph, groups))
    return FusionReport(tuple(groups), report.fused_values)


def fuse_row_gathers(graph, report):
    groups = list(report.groups)
    buckets = {}
    for group in groups:
        if (len(group.members) == 1 and group.anchor is not None
                and group.anchor.op == 'gather_rows'
                and graph.consumers_of(group.anchor.output)):
            buckets.setdefault(group.anchor.inputs[1], []).append(group)
    for candidates in buckets.values():
        start = min(graph.index_of(g.members[0]) for g in candidates)
        eligible = [g for g in candidates if all(
            graph.index_of(graph.producer_of(v)) < start
            for v in g.anchor.inputs)]
        if len(eligible) < 2:
            continue
        members = tuple(sorted((g.anchor for g in eligible), key=graph.index_of))
        groups = [g for g in groups if g not in eligible]
        groups.append(FusionGroup(members[0], members, (), row_gather=True))
        groups = list(order_by_dependencies(graph, groups))
    return FusionReport(tuple(groups), report.fused_values)
