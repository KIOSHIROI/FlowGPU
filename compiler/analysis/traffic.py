"""Symbolic logical traffic accounting and attribution (Design.md 21).

A = T_moved / T_floor compares modeled accesses with the interface convention:
each declared input is read once and each declared output written once. This
baseline is not a universal physical HBM lower bound: unused inputs, metadata-
only operands, aliases and cache residency can change actual memory traffic.
Nor does A decide runtime performance, which also depends on arithmetic,
launches, layout, scheduling and hardware resources.

Charges attribute modeled bytes to values and placement decisions. The default
counts distinct kernel operands and stores; tiled/region models additionally
estimate operand replication and sequential sweeps. Padding and real cache
behavior remain outside this model. R/B/H/D measurements are required when
comparing materialization, recomputation and tiling.
"""

import re
from dataclasses import dataclass

from compiler.ir.dataflow import INPUT, OUTPUT
from compiler.shared import group_operands
from compiler.shared import is_pointwise

# `size` is a product of the dtype's width and the dimensions, written by
# `backends.product` and `liveness.DTYPE_BYTES`. This is what it looks like;
# evaluating it at the unit shape means every dimension is 1.
IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Charge:
    """One value's share of the traffic, and what put it there."""

    value: str
    size: str
    reason: str
    count: int = 1

    @property
    def expression(self) -> str:
        if self.count == 1:
            return self.size

        return f"{self.count} * ({self.size})"


@dataclass(frozen=True)
class Traffic:
    """13.7's plan, priced: both `floor` and `moved` are modeled bytes."""

    floor: tuple
    charges: tuple

    @property
    def floor_expression(self) -> str:
        return " + ".join(entry.size for entry in self.floor) or "0"

    @property
    def moved_expression(self) -> str:
        return " + ".join(charge.expression for charge in self.charges) or "0"

    @property
    def a(self) -> float:
        """A at the unit shape.

        Every symbolic dimension is taken to be 1, which is the one shape a
        program's own text determines. It is a normalisation like any other,
        and it is the one that makes two versions of a program comparable, so
        it is stated rather than hidden.
        """
        floor = sum(at_unit(entry.size) for entry in self.floor)
        moved = sum(
            charge.count * at_unit(charge.size) for charge in self.charges
        )

        if floor == 0:
            return 1.0

        return moved / floor

    def above_the_floor(self) -> tuple:
        """The charges that are not part of the interface's own traffic."""
        at_floor = {entry.value: entry.size for entry in self.floor}

        return tuple(
            charge
            for charge in self.charges
            if charge.value not in at_floor or charge.count > 1 or charge.size != at_floor[charge.value]
        )


def at_unit(expression: str) -> int:
    """The value of a byte expression when every dimension is 1.

    The expression is always a sum of products of integers and identifiers, so
    replacing the identifiers is enough, and the check keeps that claim honest
    rather than trusting it.
    """
    substituted = IDENTIFIER.sub("1", expression)

    if not re.fullmatch(r"[0-9+\-*/\s()]+", substituted):
        raise ValueError(f"not a byte expression: {expression!r}")

    return int(eval(substituted, {"__builtins__": {}}, {}))  # noqa: S307


def _group_of(fusion) -> dict:
    """Which kernel each value is computed by."""
    owner = {}

    for number, group in enumerate(fusion.groups):
        for node in group.members:
            if node.output is not None:
                owner[node.output] = number

    return owner


def _reason(graph, fusion, value, owner) -> str:
    """Why this value is in memory rather than in registers.

    13.6's two conditions are the whole answer. An edge can be fused when one
    end of it is pointwise, and it is only fused when the value on it is read
    once -- so a value in memory is either read more than once, or it sits on
    an edge with no pointwise end, or it was read once by a pointwise operator
    that the placement rule put on the other side.
    """
    consumers = graph.consumers_of(value)
    declared_outputs = {
        node.inputs[0] for node in graph.nodes if node.op == OUTPUT
    }

    if value in declared_outputs:
        return "declared output, handed back to the caller"

    if len(consumers) > 1:
        names = ", ".join(sorted(node.output for node in consumers))

        return (
            f"read by {len(consumers)} values ({names}), so 13.6 cannot fuse "
            f"it"
        )

    if not consumers:
        return "no reader"

    readers = [
        number
        for number, group in enumerate(fusion.groups)
        if value in group_operands(graph, group)
    ]
    here = fusion.groups[owner[value]]
    there = fusion.groups[readers[0]]

    producer = graph.producer_of(value).op
    consumer = consumers[0].op

    if is_pointwise(producer) and is_pointwise(consumer):
        # Two pointwise kernels do not end up separate: the second would have
        # joined the first. Reaching here means 13.6's own rule was not
        # applied, which is a bug rather than a fact about the program.
        raise AssertionError(
            f"{value} sits between two pointwise kernels (k{owner[value]} and "
            f"k{readers[0]}), so 13.6 should have fused it"
        )

    if is_pointwise(producer) or is_pointwise(consumer):
        # Fusable, and kept in memory anyway: the pointwise operator was placed
        # on one side, so the value on the boundary is the other side's input
        # or output. Which value pays is a decision, not an accident.
        placed = (
            owner[value] if is_pointwise(producer) else readers[0]
        )

        return (
            f"fusable, but the pointwise side was placed in k{placed}, so "
            f"this value is what the boundary costs"
        )

    return (
        f"{producer} -> {consumer}: neither side is pointwise, so no edge "
        f"here can be fused"
    )


def plan_traffic(graph, fusion, memory, *, tiled=False, block_m=64, block_n=64) -> Traffic:
    """Price a plan: what the interface costs, and what the plan charges.

    Both lists follow the memory plan's own order, which follows the graph, so
    the same program always produces the same report. Building them from sets
    put the terms in hash order and made the output wobble from run to run.
    """
    if memory.stage_plan is not None:
        return _stage_traffic(graph,memory,tiled=tiled)
    sizes = {buffer.value: buffer.size for buffer in memory.buffers}
    owner = _group_of(fusion)
    declared_inputs = {
        value
        for value in graph.values
        if graph.producer_of(value).op == INPUT
    }
    declared_outputs = {
        node.inputs[0] for node in graph.nodes if node.op == OUTPUT
    }

    floor = []
    charges = []

    for buffer in memory.buffers:
        value = buffer.value
        readers = [
            number
            for number, group in enumerate(fusion.groups)
            if value in group_operands(graph, group)
        ]

        read_factor = " + ".join(tiled_read_count(graph, fusion.groups[r], value, block_m, block_n)
                                 for r in readers) if tiled else ""
        tiled_size = f"({read_factor}) * ({buffer.size})" if readers and tiled else buffer.size
        if value in declared_inputs:
            floor.append(
                Charge(
                    value=value,
                    size=buffer.size,
                    reason="declared input, read once",
                )
            )
            charges.append(
                Charge(
                    value=value,
                    size=tiled_size,
                    reason=(
                        "declared input; placement-aware kernel reads (GEMM tiles / indexed broadcasts)" if tiled else
                        "declared input, read once"
                        if len(readers) <= 1
                        else f"declared input, read by {len(readers)} kernels"
                    ),
                    count=1 if tiled else max(len(readers), 1),
                )
            )

            continue

        if value in declared_outputs:
            floor.append(
                Charge(
                    value=value,
                    size=buffer.size,
                    reason="declared output, written once",
                )
            )

        if buffer.storage is None:
            # Fused into a kernel, or read by nobody: neither moves a byte.
            continue

        # Written once by whatever produced it, then read once per kernel that
        # needs it.
        charges.append(
            Charge(
                value=value,
                size=f"({buffer.size}) + ({tiled_size})" if tiled and readers else buffer.size,
                reason=_reason(graph, fusion, value, owner),
                count=1 if tiled else 1 + len(readers),
            )
        )

    return Traffic(floor=tuple(floor), charges=tuple(charges))


def _stage_traffic(graph,memory,*,tiled):
    """Modeled full-domain touches from explicit stage ownership, not HBM.

    Repeated expression load sites, cache effects and masked padding are not
    counted. The tiled model includes base-field replication across D tiles
    and max-partial reads by every partial-producing program.
    """
    stage=memory.stage_plan
    if stage.merge is not None:
        return _merge_stage_traffic(graph,memory,tiled=tiled)
    outputs={graph.resolve(n.inputs[0]) for n in graph.nodes if n.op==OUTPUT}
    floor=[];charges=[]
    tiles=stage.d_tiles if tiled else '1'
    segments=stage.segments if tiled else '1'
    for buffer in memory.buffers:
        value=buffer.value
        if graph.producer_of(value).op==INPUT:
            floor.append(Charge(value,buffer.size,'declared input, read once'))
            factors=[]
            if value in stage.stages[0].inputs:factors.append('1')
            if value in stage.stages[1].inputs:factors.append(tiles if value in stage.base_sources else '1')
            if factors:charges.append(Charge(value,f"({' + '.join(factors)}) * ({buffer.size})",
                'segmented stage input scans; base fields replicated across D tiles' if tiled else 'segmented stage input reads'))
        elif value in outputs:
            floor.append(Charge(value,buffer.size,'declared output, written once'))
            charges.append(Charge(value,buffer.size,'segmented epilogue writes declared output'))
    for index,scratch in enumerate(stage.scratch):
        reads=f'(({segments}) + 1) * ({tiles})' if index==0 else tiles if index==1 else '1'
        charges.append(Charge(scratch.name,f'(1 + ({reads})) * ({scratch.size})',
            f'stage-private scratch; written by stage{scratch.writer}, read by {scratch.readers}; logical touches'))
    return Traffic(tuple(floor),tuple(charges))


def _merge_stage_traffic(graph,memory,*,tiled):
    """Full-domain touch model, including a conservative exceptional replay.

    Conditional charges are separately labeled so explanations can expose
    both the ordinary path and the upper bound. Neither is physical HBM.
    """
    stage=memory.stage_plan;tiles=stage.d_tiles if tiled else '1'
    outputs={graph.resolve(n.inputs[0]) for n in graph.nodes if n.op==OUTPUT}
    floor=[];charges=[]
    for buffer in memory.buffers:
        value=buffer.value
        if graph.producer_of(value).op==INPUT:
            floor.append(Charge(value,buffer.size,'declared input, read once'))
            if value in stage.stages[0].inputs:
                base=value in stage.base_sources
                normal=f'2 * ({tiles})' if base else '1'
                replay=tiles if base else '1'
                charges.append(Charge(value,f'({normal}) * ({buffer.size})','local statistics input scans; max/weights replicated across D tiles'))
                charges.append(Charge(value,f'({replay}) * ({buffer.size})','conditional replay upper bound; original expression for exceptional tiles'))
        elif value in outputs:
            floor.append(Charge(value,buffer.size,'declared output, written once'))
            charges.append(Charge(value,buffer.size,'merged epilogue writes declared output'))
    for index,scratch in enumerate(stage.scratch):
        reads=tiles if index<2 else '1'
        charges.append(Charge(scratch.name,f'(1 + ({reads})) * ({scratch.size})',
            f'stage-private scratch; written by stage{scratch.writer}, read by {scratch.readers}; logical touches'))
    return Traffic(tuple(floor),tuple(charges))


def ratio_at(traffic, sizes) -> float:
    """A at a concrete shape, for a caller that knows one.

    The symbolic form is what the compiler can always produce; this is for the
    case where a caller has chosen the dimensions and wants to compare two
    programs at the same size.
    """
    def evaluate(expression):
        substituted = expression
        for dimension, value in sizes.items():
            substituted = re.sub(
                rf"\b{re.escape(dimension)}\b", str(value), substituted
            )

        return at_unit(substituted)

    floor = sum(evaluate(entry.size) for entry in traffic.floor)
    moved = sum(
        charge.count * evaluate(charge.size) for charge in traffic.charges
    )

    if floor == 0:
        return 1.0

    return moved / floor





def placement_cost(block_m=64, block_n=64):
    """Local change in logical global-memory bytes, with no cache prediction.

    Literal shapes only: a symbolic dimension gives no basis for deciding
    which materialization is cheaper, so ties preserve the rule policy.
    This minimizes this additive placement model, not measured latency.
    """
    from compiler.analysis.precision import DTYPE_BYTES
    from math import prod, ceil

    def size(graph, value):
        tensor = graph.values[value]
        if not all(str(d).isdigit() for d in tensor.shape):
            return None
        return prod(map(int, tensor.shape)) * DTYPE_BYTES[tensor.dtype]

    def cost(graph, members, anchor, side):
        if anchor is None:
            return 0
        inside = {n.output for n in members}
        if side == "epilogue":
            amount = size(graph, anchor.output)
            return -2 * amount if amount is not None else 0
        outputs = [v for v in anchor.inputs if v in inside]
        external = {v for n in members for v in n.inputs if v not in inside}
        saved = [size(graph, v) for v in outputs]
        reads = [size(graph, v) for v in external]
        if any(v is None for v in saved + reads):
            return 0
        repeat = 1
        if anchor.op == "matmul":
            left, right = anchor.inputs
            m, k = graph.values[left].shape
            _, n = graph.values[right].shape
            if not str(m).isdigit() or not str(n).isdigit():
                return 0
            repeat = max([ceil(int(n)/block_n) if v == left else ceil(int(m)/block_m)
                          for v in outputs] or [1])
        return (repeat - 1) * sum(reads) - (repeat + 1) * sum(saved)
    return cost


def tiled_read_count(graph, group, value, block_m=64, block_n=64):
    """GEMM tile reuse, including prologue leaves; counts logical loads.

    HBM cache hits are deliberately not inferred from this number.
    Returns a symbolic expression when a dimension is not literal.
    """
    if group.indexed:
        return indexed_read_count(graph, group, value)
    anchor = group.anchor
    if anchor is None or anchor.op != "matmul":
        return "1"
    def leaves(v):
        node = next((n for n in group.prologue if n.output == v), None)
        return set().union(*(leaves(x) for x in node.inputs)) if node else {v}
    left, right = anchor.inputs
    factors = []
    for operand, dim, block in [(left, graph.values[right].shape[1], block_n),
                                (right, graph.values[left].shape[0], block_m)]:
        if value in leaves(operand):
            factors.append(str((int(dim)+block-1)//block) if str(dim).isdigit()
                           else f"(({dim} + {block-1}) // {block})")
    # The same pointer on both sides is loaded for each side.
    return " + ".join(factors) or "1"


def indexed_read_count(graph, group, value):
    """Logical replication through broadcast paths, without cache prediction.

    Permutations/reshape visit the same number of elements. A broadcast leaf
    is read once per expanded coordinate, even when no expanded tensor exists.
    Constants and dtype templates need no data; for metadata-only arguments
    retain the older conservative interface charge of one. Arbitrary gather
    still uses that conservative full-input charge, not a unique-index model.
    """
    if group.cross_domain:
        return cross_domain_read_count(graph,group,value)
    if group.reduction_domain:
        return domain_read_count(graph, group, value)
    from compiler.shared import product, group_results
    produced = {n.output: n for n in group.members}
    paths = {}

    def visit(v, remaps=(), repeats=()):
        node = produced.get(v)
        if node is None:
            if v == value:
                paths[remaps] = ' * '.join(repeats) or '1'
            return
        if node.op in {'ones_like', 'zeros_like', 'neg_inf_like'}:
            return
        inputs = node.inputs
        if node.op in {'reshape', 'cast_like', 'cast_to', 'to_fp32', 'broadcast_like', 'move_first_last', 'sum_last', 'max_last'}:
            inputs = inputs[:1]
        if node.op == 'broadcast_like':
            ratio = f'(({product(graph.values[v].shape)}) // ({product(graph.values[inputs[0]].shape)}))'
            visit(inputs[0], remaps + (v,), repeats + (ratio,))
        else:
            mapped = remaps + (v,) if node.op in {'move_first_last', 'sum_last', 'max_last', 'gather_row_offset'} else remaps
            for i, operand in enumerate(inputs):
                # The gather index is evaluated in the output index space;
                # only the gathered data uses the indirect-address path.
                visit(operand, remaps if node.op == 'gather_row_offset' and i == 1 else mapped, repeats)
    for result in group_results(graph, group):
        visit(result)
    return ' + '.join(paths.values()) or '1'


def domain_read_count(graph, group, value):
    """Count sequential column sweeps; cached row statistics are not re-read
    for every broadcast coordinate. Metadata keeps conservative interface cost.
    """
    from compiler.shared import product, group_results
    from compiler.shared import row_key
    from compiler.shared import REDUCTIONS
    produced={n.output:n for n in group.members}
    row=row_key(group.reduction_domain[:-1])
    is_row=lambda v:row_key(graph.values[v].shape)==row
    reductions=[n for n in group.members if n.op in REDUCTIONS]
    results=group_results(graph,group)
    unary={'reshape','cast_like', 'cast_to','to_fp32','broadcast_like','move_first_last'}
    constants={'ones_like','zeros_like','neg_inf_like'}

    def paths(roots, row_scope=False):
        seen=set(); found=set()
        def visit(v, remaps=()):
            if (v,remaps) in seen:return
            seen.add((v,remaps))
            n=produced.get(v)
            if not row_scope and is_row(v):return
            if n is None:
                if v==value:found.add(remaps)
                return
            if n.op in REDUCTIONS or n.op in constants:return
            inputs=n.inputs[:1] if n.op in unary else n.inputs
            mapped=remaps+(v,) if n.op in {'move_first_last','broadcast_like'} else remaps
            for x in inputs:visit(x,mapped)
        for root in roots:visit(root)
        return found

    # Row leaves are loaded once outside the column loops.
    row_reads=paths([n.output for n in group.members if is_row(n.output) and n.op not in REDUCTIONS],True)
    if is_row(value):row_reads.add(()) # External row operands are preloaded.
    factors=[]
    if row_reads:
        factors.append(f'(({product(group.reduction_domain[:-1])}) // ({product(graph.values[value].shape)}))')
    stored=set()
    def contains(root,target):
        if root==target:return True
        n=produced.get(root)
        return bool(n and n.op not in REDUCTIONS and n.op not in constants and
                    any(contains(x,target) for x in (n.inputs[:1] if n.op in unary else n.inputs)))
    owners={v:next((r.output for r in reversed(reductions) if contains(r.inputs[0],v)),None)
            for v in results if not is_row(v)}
    roots_by_sweep=[]
    by_output={r.output:r for r in reductions}
    batches=group.reduction_batches or tuple((r.output,) for r in reductions)
    for batch in batches:
        exported=[v for v,owner in owners.items() if owner in batch]
        stored.update(exported)
        roots_by_sweep.append([by_output[v].inputs[0] for v in batch]+exported)
    remaining=[v for v in results if not is_row(v) and v not in stored]
    if remaining:roots_by_sweep.append(remaining)
    for roots in roots_by_sweep:
        reads=len(paths(roots))
        if reads:factors.append(f'({reads}) * (({product(group.reduction_domain)}) // ({product(graph.values[value].shape)}))')
    return ' + '.join(factors) or '1'


def cross_domain_read_count(graph,group,value):
    """Logical sweep reads for opt-in R/D recomputation (not measured HBM).

    Base-domain inputs repeat per D tile in each reduction prologue; expanded
    inputs span D once. Reduction outputs and metadata do not read memory.
    """
    from compiler.shared import product
    from compiler.optimization.indexed_fusion import REDUCTIONS
    from compiler.optimization.reduction_fusion import row_key
    produced={n.output:n for n in group.members}
    base_row=row_key(group.cross_domain[:-1]);d=group.reduction_domain[-2];dt=group.cross_tile[1]
    copies=f'(({d} + {dt} - 1) // {dt})'
    def leaves(v):
        n=produced.get(v)
        if n is None:return {v}
        if n.op in REDUCTIONS or n.op in {'ones_like','zeros_like','neg_inf_like'}:return set()
        inputs=n.inputs[:1] if n.op in {'reshape','cast_like', 'cast_to','to_fp32','broadcast_like','move_first_last'} else n.inputs
        return set().union(*(leaves(x) for x in inputs))
    base_reductions=[n for n in group.members if n.op in REDUCTIONS and n is not group.anchor]
    base_inputs=set().union(*(leaves(n.inputs[0]) for n in base_reductions))
    wide_inputs=leaves(group.anchor.inputs[0])
    if value in base_inputs:
        if row_key(graph.values[value].shape)==base_row:return copies
        reductions={n.output:n for n in base_reductions+[group.anchor]}
        batches=group.reduction_batches or tuple((v,) for v in reductions)
        passes=sum(value in set().union(*(leaves(reductions[v].inputs[0]) for v in batch)) for batch in batches)
        return f'{passes} * ({copies})'
    if value in wide_inputs:return '1'
    return '1' # dtype/shape templates retain conservative interface cost
