"""Conservative kernel fusion heuristic.

Same-index pointwise regions may join one anchor when the connecting value
has a single consumer and every external operand exists before kernel execution.
This is a compiler policy, not a necessary-and-sufficient hardware fusion theorem.
RoPE uses pair indexing and is an anchor; RMSNorm -> RoPE stays separate.
"""

from dataclasses import dataclass

from flowgpu.compiler.ir.dataflow import INPUT, OUTPUT, DataflowGraph, Node
from flowgpu.compiler.core.operators import OPERATORS  # noqa: F401  (registry contract)
from flowgpu.compiler.shared import is_pointwise  # noqa: F401  (re-exported here)


def is_operator(node: Node) -> bool:
    """`input` and `output` are structure, not computation (13.4)."""
    return node.op not in (INPUT, OUTPUT)


@dataclass(frozen=True)
class FusionGroup:
    """The operators one kernel evaluates, in topological order.

    `anchor` is the non-pointwise operator the group is built around. It is
    None only when the group is pointwise throughout: still one kernel, just
    one that reads both operands from memory.

    `prologue` and `epilogue` say on which side of the anchor the pointwise
    operators sit, which is what 13.6's placement decided: a prologue region
    produces one of the anchor's operands and is computed on the way in, an
    epilogue region consumes the anchor's result and is applied to it. A group
    can have both (`silu` into a matmul's operand and `relu` onto its result),
    which is why this is recorded rather than assumed.

    `fused_values` are the values produced and consumed inside the group.
    They never need a buffer (11: "materialization requirement").
    """

    anchor: Node | None
    members: tuple
    fused_values: tuple
    prologue: tuple = ()
    epilogue: tuple = ()
    # A shape-aware expression region, emitted by the indexed backend rather
    # than the same-index pointwise/anchor templates.
    indexed: bool = False
    # A shared last-axis domain with row and full-domain outputs. Reductions
    # execute sequential bounded sweeps inside a single program, per row tile.
    reduction_domain: tuple = ()
    # Ordered batches of independent reduction output names. Both emission
    # and traffic accounting consume this schedule, rather than re-derive it.
    reduction_batches: tuple = ()
    # Optional opt-in recomputation across one inserted broadcast axis.
    # Base statistics are repeated per expanded-axis tile, never per lane.
    cross_domain: tuple = ()
    cross_tile: tuple = ()
    # Explicit sibling gathers with one shared index input and multiple results.
    # No value is eliminated; the optimization removes launch boundaries.
    row_gather: bool = False
    # Explicit local mapping for a selected gather partition; no policy default.
    row_gather_schedule: tuple = ()

    @property
    def fused(self) -> bool:
        """More than one operator in one kernel: an actual fusion."""
        return len(self.members) > 1


@dataclass(frozen=True)
class FusionReport:
    groups: tuple
    fused_values: frozenset

    @property
    def fusions(self) -> tuple:
        return tuple(group for group in self.groups if group.fused)


MULTI_RESULT_OPS = ('store_cache_4d', 'linear_decode_forward', 'topk_last',
                    'segment_rank', 'gated_delta_recurrence',
                    'argmax_last', 'argmin_last',
                    # The carried-tuple scans: values and indices are one
                    # computation, so the two result nodes must share a kernel
                    # (P1 of docs/ITERATION_REGION.md).
                    'cummin_last', 'cummax_last',
                    # `cumsum_totals` is a scan whose two results are one kernel:
                    # the scan and the carry it already computes.
                    'cumsum_totals')


def order_by_dependencies(graph, groups, *, index_of=None) -> tuple:
    """Groups in launch order: every producer runs before its consumers.

    Ordering by the first member's index is not enough.  A fused pointwise
    region can consume a value defined *after* its own first member, and then
    the producer's group is launched later than the consumer -- the consumer
    reads a buffer nobody has written yet.  Reproduced 2026-10-24 with an
    integer-only FP8 encoder (the saturation chain was scheduled before the
    `e4c` group it reads, and the reference backend consumed `float('nan')`).

    This is a **stable** topological sort: the previous order is the tie-break,
    so a program whose groups were already in dependency order keeps its
    byte-identical kernel sequence.  Every pass that rebuilds or reorders groups
    ends with this call, so no pass can silently undo it.
    """
    index_of = index_of or graph.index_of
    groups = sorted(groups, key=lambda group: index_of(group.members[0]))
    producer_of = {}
    for key, group in enumerate(groups):
        for node in group.members:
            if node.output is not None:
                producer_of[node.output] = key
    dependencies = [set() for _ in groups]
    for key, group in enumerate(groups):
        inside = {node.output for node in group.members}
        for node in group.members:
            for value in node.inputs:
                if value in inside:
                    continue
                other = producer_of.get(value)
                if other is not None and other != key:
                    dependencies[key].add(other)
    ordered = []
    done = set()
    pending = list(range(len(groups)))
    while pending:
        for position, key in enumerate(pending):
            if dependencies[key] <= done:
                ordered.append(groups[key])
                done.add(key)
                pending.pop(position)
                break
        else:
            # Only an IR with a cycle could reach this; keep the remaining
            # original order rather than silently dropping kernels.
            ordered.extend(groups[key] for key in pending)
            break
    return tuple(ordered)


def coalesce_store_cache_groups(graph, report):
    """Join the SSA result nodes of a multi-result operator into one kernel.

    A multi-result operator has one dataflow node per result, so fusion sees
    sibling groups that must become one kernel: a cache store's K and V writes,
    a decode's output and state, and `topk_last`'s values and indices are each
    computed once and cannot be emitted twice without computing them twice.

    With scans this is no longer always "one member per group": the running
    result may carry a fused pointwise epilogue (`add_broadcast(scan, row)`),
    while the carried-totals sibling sits in a group of its own.  The merge
    therefore keeps every member and every fused prologue/epilogue, and unions
    `fused_values` -- the intermediate values a member consumes are registers of
    the merged kernel, not buffers.
    """
    groups = list(report.groups)
    by_key = {}
    for index, group in enumerate(groups):
        if group.anchor is None or group.anchor.op not in MULTI_RESULT_OPS:
            continue
        key = (group.anchor.op, tuple(group.anchor.inputs))
        by_key.setdefault(key, []).append(index)
    for indices in by_key.values():
        if len(indices) < 2:
            continue
        index_of = graph.index_of
        members = sorted((node for i in indices for node in groups[i].members),
                         key=lambda node: (node.part is None, node.part or 0,
                                           index_of(node)))
        inside = {id(node) for node in members}
        results = {
            node.output
            for node in members
            if node.output is not None
            and not all(id(consumer) in inside
                        for consumer in graph.consumers_of(node.output))
        }
        fused = []
        for node in members:
            if node.output is None or node.output in results:
                continue
            if any(id(consumer) in inside and consumer is not node
                   for consumer in graph.consumers_of(node.output)):
                fused.append(node.output)
        prologue = sorted((node for i in indices for node in groups[i].prologue),
                          key=index_of)
        epilogue = sorted((node for i in indices for node in groups[i].epilogue),
                          key=index_of)
        merged = FusionGroup(anchor=members[0], members=tuple(members),
                             fused_values=tuple(fused),
                             prologue=tuple(prologue), epilogue=tuple(epilogue))
        for i in sorted(indices, reverse=True):
            groups.pop(i)
        groups.append(merged)
    return FusionReport(groups=order_by_dependencies(graph, groups),
                        fused_values=report.fused_values)


def fuse(graph: DataflowGraph, *, opaque_anchors=("attention",), placement_cost=None) -> FusionReport:
    """Partition the graph's operators into the kernels that evaluate them."""
    graph.validate()
    index_of = graph.index_of
    by_index = {index_of(node): node for node in graph.nodes}

    single_use = {
        value: len(graph.consumers_of(value)) == 1 for value in graph.values
    }

    parent = {}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(one, other):
        parent[find(other)] = find(one)

    # 1. Chains of pointwise operators that could be evaluated together. Two
    #    pointwise operators only join when the value between them is read
    #    once, because that value is the one the fusion keeps in registers.
    for node in graph.nodes:
        if is_operator(node) and is_pointwise(node.op):
            parent[index_of(node)] = index_of(node)

    for edge in graph.edges():
        producer, consumer = edge.producer, edge.consumer

        if not (is_operator(producer) and is_operator(consumer)):
            continue
        if not (is_pointwise(producer.op) and is_pointwise(consumer.op)):
            continue
        if not single_use[edge.value]:
            continue

        union(index_of(producer), index_of(consumer))

    regions = {}
    for key in parent:
        regions.setdefault(find(key), []).append(by_index[key])

    anchor_keys = {
        index_of(node)
        for node in graph.nodes
        if is_operator(node) and not is_pointwise(node.op)
    }

    # 2. Place each region in exactly one anchor: the producer when there is
    #    a choice, so the pointwise work is an epilogue of the kernel that
    #    produced its operand, and the consumer otherwise, so a pointwise
    #    operand of a kernel is computed on the way in. 11's MatMul + ReLU is
    #    the first; 12.1's Add -> RMSNorm is the second, because Add has no
    #    operator producer to join.
    placement = {}
    for root, members in regions.items():
        inside = {index_of(node) for node in members}

        incoming = []
        outgoing = []

        for node in members:
            for value in node.inputs:
                producer = graph.producer_of(value)

                if producer is not None and index_of(producer) not in inside:
                    incoming.append((value, index_of(producer)))

            if node.output is None:
                continue

            for consumer in graph.consumers_of(node.output):
                if index_of(consumer) not in inside:
                    outgoing.append((node.output, index_of(consumer)))

        anchor = None
        side = None
        # A fused kernel executes at its earliest member. All external
        # operands must already exist then. Prefer the latest producer:
        # SwiGLU must compute the other GEMM before its fused mul epilogue.
        def available(key):
            start = min(key, min(inside))
            return all(producer < start for _, producer in incoming if producer != key)

        choices = [(None, None)]
        for value, key in sorted(incoming, key=lambda pair: pair[1], reverse=True):
            if key in anchor_keys and by_index[key].op not in opaque_anchors and single_use[value] and available(key):
                choices.append((key, "epilogue"))
                anchor = key
                # The anchor produces this region's operand, so the region is
                # evaluated after it: an epilogue.
                side = "epilogue"
                if placement_cost is None:
                    break

        if anchor is None or placement_cost is not None:
            for value, key in sorted(outgoing, key=lambda pair: pair[1]):
                if key in anchor_keys and by_index[key].op not in opaque_anchors and single_use[value] and available(key):
                    choices.append((key, "prologue"))
                    anchor = key
                    # The anchor reads this region's result, so the region is
                    # evaluated before it: a prologue.
                    side = "prologue"
                    if placement_cost is None:
                        break

        if placement_cost is not None:
            # Keep a deterministic tie preference: epilogue, prologue, materialize.
            anchor, side = min(choices, key=lambda choice: (
                placement_cost(graph, members, by_index.get(choice[0]), choice[1]),
                {"epilogue": 0, "prologue": 1, None: 2}[choice[1]]))
        placement[root] = (anchor, side)

    # 3. One group per anchor, plus one for every region that joined nothing.
    #    An anchor that no region joined is still a kernel -- it is the
    #    operator the backend has to emit -- so it gets a group of its own.
    #    Only anchors a region actually joined are known from the regions, so
    #    the rest are added here; leaving them out silently dropped every
    #    unfused non-pointwise operator from the partition.
    grouped = {}
    for root, members in regions.items():
        anchor, side = placement[root]

        if anchor is None:
            grouped.setdefault(("alone", root), []).append((members, None))
        else:
            grouped.setdefault(("anchor", anchor), []).append((members, side))

    for key in anchor_keys:
        grouped.setdefault(("anchor", key), [])

    groups = []
    for kind in grouped:
        anchor = by_index[kind[1]] if kind[0] == "anchor" else None

        members = [anchor] if anchor is not None else []
        prologue = []
        epilogue = []

        for region, side in grouped[kind]:
            members.extend(region)

            if side == "prologue":
                prologue.extend(region)
            elif side == "epilogue":
                epilogue.extend(region)

        members.sort(key=index_of)
        prologue.sort(key=index_of)
        epilogue.sort(key=index_of)
        inside = {index_of(node) for node in members}

        fused_values = tuple(
            node.output
            for node in members
            if node.output is not None
            and any(
                index_of(consumer) in inside
                for consumer in graph.consumers_of(node.output)
            )
        )

        groups.append(
            FusionGroup(
                anchor=anchor,
                members=tuple(members),
                fused_values=fused_values,
                prologue=tuple(prologue),
                epilogue=tuple(epilogue),
            )
        )

    # See `order_by_dependencies` for why this is not a plain index sort.
    return FusionReport(
        groups=order_by_dependencies(graph, groups),
        fused_values=frozenset(
            value for group in groups for value in group.fused_values
        ),
    )
