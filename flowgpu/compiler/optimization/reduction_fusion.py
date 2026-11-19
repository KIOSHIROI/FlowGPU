"""Shared-domain, multi-output reduction regions; no cross-domain cloning.

By default at most two ordered reductions share a last-axis domain. Full-domain results
used by other kernels remain materialized (e.g. weights reused across D), but
are written by the reduction sweep instead of a separate kernel.
"""
from flowgpu.compiler.optimization.fusion import (FusionGroup, FusionReport,
                                           order_by_dependencies)
from flowgpu.compiler.optimization.indexed_fusion import REDUCTIONS, supported
from flowgpu.compiler.shared import ROW_WISE, full_key, row_key  # noqa: F401

# Pure same-shape elementwise operators a sweep may recompute at a mapped
# address when they sit inside a foldable chain. One mapped index serves
# every operand, so per-element math commutes with the rotation maps.
ELEMENTWISE_LINKS = frozenset({
    'cast_like', 'cast_to', 'to_fp32', 'add', 'sub', 'mul', 'div', 'relu', 'silu',
    'fused_sigmoid_mul', 'exp', 'log', 'isfinite',
})


def reduction_batches(graph, members, domain, *, coalesce=False):
    """Schedule reductions together only when their row dependencies are ready.

    A computed row expression is a barrier. Full-domain expressions between
    independent reductions are evaluated once in the common sweep instead.
    """
    produced = {n.output: n for n in members}
    row = row_key(domain[:-1])
    is_row = lambda v: row_key(graph.values[v].shape) == row
    unary = {'reshape', 'cast_like', 'cast_to', 'to_fp32', 'broadcast_like', 'move_first_last'}
    constants = {'ones_like', 'zeros_like', 'neg_inf_like'}
    def requirements(value):
        if is_row(value): return {value}
        node = produced.get(value)
        if node is None or node.op in constants: return set()
        inputs = node.inputs[:1] if node.op in unary else node.inputs
        return set().union(*(requirements(v) for v in inputs))
    ready = {v for n in members for v in n.inputs if v not in produced and is_row(v)}
    batches, pending = [], []
    def flush():
        if pending:
            batches.append(tuple(pending)); ready.update(pending); pending.clear()
    for node in members:
        if node.op in REDUCTIONS:
            if pending and (not coalesce or not requirements(node.inputs[0]) <= ready):
                flush()
            pending.append(node.output)
        elif is_row(node.output):
            flush(); ready.add(node.output)
    flush()
    return tuple(batches)


def fuse_reductions(graph, report, *, max_reductions=2, coalesce=False):
    if type(max_reductions) is not int or max_reductions <= 0:
        raise ValueError('max_reductions must be a positive integer')
    from flowgpu.compiler.shared import group_results, store_rotation_chain
    groups = list(report.groups)
    index = graph.index_of
    for seed in list(groups):
        if seed not in groups:
            continue
        # Row-wise anchors (softmax, rms_norm) own the same shared row
        # statistics and the same sweep emitter, so they seed the merge too;
        # their output rotations are the store-side case.
        reductions = [n for n in seed.members if n.op in REDUCTIONS]
        anchors = [n for n in seed.members
                   if n.op in ROW_WISE and not reductions]
        if not reductions and not anchors:
            continue
        if reductions:
            domain = tuple(graph.values[reductions[0].inputs[0]].shape)
        else:
            domain = tuple(graph.values[anchors[0].inputs[0]].shape)
        full, row = full_key(domain), row_key(domain[:-1])
        owner = {n.output:i for i,g in enumerate(groups) for n in g.members}

        def rotates_reduction_output(n):
            """True when a rotation chain roots at this seed's reduction.

            The seed's anchor may be softmax or rms_norm as well as the
            sum/max reductions; those anchors own the shared row statistics
            the same way, and the store side only readdresses output memory.
            """
            if n.op != 'move_first_last':
                return False
            root = n.inputs[0]
            while True:
                producer = graph.producer_of(root)
                if producer is None or producer.op == 'input':
                    return False
                if producer.op in REDUCTIONS or producer.op in ROW_WISE:
                    return True
                if producer.op != 'move_first_last':
                    return False
                root = producer.inputs[0]

        def fusable_rotation(n):
            """Load-side rotations of inputs and store-side rotations of the
            reduction output are both foldable; a rotation of any other
            computed value may cross rows and stays materialized."""
            if n.op != 'move_first_last':
                return True
            producer = graph.producer_of(n.inputs[0])
            return rotates_reduction_output(n) or \
                producer is None or producer.op == 'input'

        def eligible(group):
            # A candidate that is itself a foldable chain is checked as a
            # whole: its rotation intermediates are pure readdressing and
            # its elementwise links are recomputed at the mapped address,
            # so their shapes need not fit the domain keys -- the sweep
            # folds the entire chain into addressing and per-element math.
            if group.members and all(
                    n.op == 'move_first_last'
                    or n.op in ELEMENTWISE_LINKS
                    for n in group.members):
                return True
            for n in group.members:
                if not supported(n.op) or n.op in {'gather_row_offset', 'gather_rows'}:
                    return False
                shape = graph.values[n.output].shape
                if row_key(shape) != row and full_key(shape) != full \
                        and not rotates_reduction_output(n):
                    return False
                if n.op in REDUCTIONS and full_key(graph.values[n.inputs[0]].shape) != full:
                    return False
                if n.op == 'broadcast_like':
                    source = graph.values[n.inputs[0]].shape
                    if row_key(source) == row and row_key(shape) != row:
                        padded = ['1'] * (len(shape)-len(source)) + list(source)
                        # Equal numeric dimensions do not prove row semantics:
                        # [2,2] -> [2,2,2] aligns to columns, not the first rows.
                        if str(padded[-1]) != '1' or row_key(padded[:-1]) != row:
                            return False
                # Permuting a computed value may cross rows. The shared row
                # statistics cannot be substituted under that remap. The two
                # exceptions are load-side rotations of pipeline inputs (the
                # emitter readdresses loads) and store-side rotations of the
                # reduction's own output (the sweep readdresses stores).
                # Chained rotations of the reduction output are exactly the
                # inverse-rotation tail that *_axis normalization produces.
                if n.op == 'move_first_last' and not fusable_rotation(n):
                    return False
                if n.op == 'move_first_last' and not rotates_reduction_output(n):
                    source = graph.values[n.inputs[0]].shape
                    if row_key(source) == row and str(source[0]) != '1' and len(row) > 1:
                        return False
            return True

        # The sweep emitter owns reductions and row-wise anchors directly, so
        # the seed need not be an indexed-fusion operator. Candidates still
        # must be, so unknown anchors never join a sweep region. A seed that
        # the strict gate already accepts keeps the old strict semantics --
        # its own outputs fitting the domain keys is what let bare
        # reductions merge before, and shapes like a max over the last axis
        # of [2,2,2] (result [2,2]) deliberately do not fit: their row
        # statistics do not survive the equal-dimension broadcast trap.
        # Only a row-wise anchor the strict gate rejects gets the relaxed
        # gate, and that relaxation admits nothing but the anchor itself,
        # input rotations, and the anchor's own store-rotation tail.
        def relaxed_member(n):
            return (n.op in ROW_WISE
                    or n.op == 'move_first_last' and fusable_rotation(n))

        # The seed's own anchor decides the library contract: a library
        # anchor must stay unfused (see the admission note below).
        seed_anchor = next(
            (n for n in seed.members if n.op in ROW_WISE), None
        )
        if seed_anchor is None:
            seed_anchor = next(
                (n for n in seed.members if n.op in REDUCTIONS), None
            )

        strict = eligible(group=seed)
        if not strict and not all(
                relaxed_member(n) or supported(n.op) for n in seed.members):
            continue

        # 13.7's contract, checked on the MERGED group: a pointwise member
        # may only join the sweep when the sweep is its sole reader. A value
        # that escapes the merged group (an output, or another kernel's
        # operand) keeps its buffer and its kernel, exactly as when the seed
        # was a bare reduction. Anchor outputs and the store-rotation tail
        # are the sweep's products, not absorbed readers, so they are exempt.
        selected = {groups.index(seed)}
        count = len(reductions)
        changed = True
        while changed:
            changed = False
            members = [n for i in selected for n in groups[i].members]
            neighbors = set()
            for n in members:
                neighbors.update(owner[v] for v in n.inputs if v in owner)
                neighbors.update(owner[c.output] for c in graph.consumers_of(n.output) if c.output in owner)
            for i in sorted(neighbors - selected):
                candidate = groups[i]
                added = sum(n.op in REDUCTIONS for n in candidate.members)
                # A relaxed seed absorbs only what the sweep can fold:
                # same-domain reductions, chains rooted at the anchor's
                # output (the store-rotation tail), and chains rooted at a
                # pipeline input (the load-side 13.6 fold). A chain may
                # interleave pure elementwise links -- a cast between the
                # input and the first rotation, a multiply feeding the
                # chain -- because the sweep recomputes them elementwise at
                # the mapped address, exactly as it recomputes any 13.7
                # prologue. Every non-rotation link's output must be read
                # only inside the merged sweep (enforced by `consumable`),
                # and each elementwise link must be same-shape elementwise
                # so one mapped address serves every operand.
                # One exception stands: a library anchor's kernel form is
                # opaque, so rms_norm/rmsnorm_hf seeds absorb rotations
                # only -- never an elementwise link, which would make the
                # region "fused" and break the library dispatch contract.
                library_seed = seed_anchor is not None and \
                    seed_anchor.op in {'rms_norm', 'rmsnorm_hf'}

                def is_foldable_chain(cand):
                    if library_seed:
                        return False
                    links = {}
                    for n in cand.members:
                        if n.op not in {'move_first_last'} | ELEMENTWISE_LINKS:
                            return False
                        links[n.output] = n
                    anchor_roots = REDUCTIONS | set(ROW_WISE)

                    def chain_root(value, visited=()):
                        """The chain's external source, walking its links.

                        A value is the root when nothing in the pipeline
                        produces it (a pipeline input) or its producer is
                        the sweep's own anchor (the store side); a producer
                        that is itself a link of this chain just extends
                        the walk.
                        """
                        producer = graph.producer_of(value)
                        if producer is None or producer.op == 'input' \
                                or producer.op in anchor_roots:
                            return value
                        if producer.output in links and \
                                index(producer) not in visited:
                            return chain_root(producer.inputs[0],
                                              visited + (index(producer),))
                        return None

                    root = None
                    for n in cand.members:
                        for value in n.inputs:
                            if value not in links:
                                found = chain_root(value)
                                if found is not None:
                                    root = value
                    if root is None:
                        return False
                    seen = set()
                    while True:
                        node = next((m for m in cand.members
                                     if root in m.inputs
                                     and id(m) not in seen), None)
                        if node is None:
                            break
                        if id(node) in seen:
                            return False
                        seen.add(id(node))
                        # An elementwise link must be same-shape
                        # elementwise so one mapped address serves every
                        # operand. External readability is not checked
                        # here: `consumable` already enforces 13.7's
                        # register contract against the MERGED region, and
                        # a link whose only readers sit in the seed group
                        # is exactly the case a candidate-local test would
                        # wrongly refuse.
                        if node.op != 'move_first_last':
                            source_shape = graph.values[node.inputs[0]].shape
                            for value in node.inputs[1:]:
                                if graph.values[value].shape != source_shape:
                                    return False
                        root = node.output
                    return all(id(n) in seen for n in cand.members)

                consumable = strict or all(
                    n.op in REDUCTIONS or n.op in ROW_WISE
                    or n.op == 'move_first_last'
                    or all(c.op != 'output' and owner.get(c.output, i) in
                           (neighbors | selected | {i})
                           for c in graph.consumers_of(n.output))
                    for n in candidate.members
                )
                foldable = strict or added > 0 \
                    or is_foldable_chain(candidate)
                # Scheduling feasibility is a per-candidate admission test,
                # not a post-merge veto: admitting a candidate whose
                # producer runs after the region's first member (or that
                # pushes the first member earlier than a producer an
                # already-selected member needs) would read a value before
                # the kernel that computes it has run (the late-matmul
                # hazard). Skip that candidate; the rest of the region can
                # still form.
                trial = [n for i in selected | {i} for n in groups[i].members]
                trial_start = min(index(n) for n in trial)
                trial_inside = {index(n) for n in trial}
                schedulable = not any(
                    index(p) >= trial_start
                    for n in trial for v in n.inputs
                    if (p := graph.producer_of(v)) is not None
                    and index(p) not in trial_inside
                )
                if count + added <= max_reductions and eligible(candidate) \
                        and consumable and foldable and schedulable:
                    selected.add(i); count += added; changed = True
        if len(selected) == 1:
            continue
        members = sorted((n for i in selected for n in groups[i].members),key=index)
        inside = {index(n) for n in members}
        anchor = next((n for n in members if n.op in REDUCTIONS), None)
        if anchor is None:
            anchor = next(n for n in members if n.op in ROW_WISE)
        # The sweep's own products: the anchor output, the store-rotation
        # tail, and every group result (they are stored, not re-read).
        from flowgpu.compiler.shared import group_results as _results, \
            store_rotation_chain as _chain
        probe = FusionGroup(anchor, tuple(members), (), (), (), indexed=True,
                            reduction_domain=domain)
        products = {n.output for n in members
                    if n.op in REDUCTIONS or n.op in ROW_WISE}
        products |= {node.output for node in _chain(graph, probe)}
        products |= set(_results(graph, probe))
        if any(n.output not in products
               and any(index(c) not in inside
                       for c in graph.consumers_of(n.output))
               for n in members):
            continue
        # A member may not depend on a producer scheduled after the region's
        # first member: fusing across that ordering would read a value
        # before the kernel that computes it has run (the late-matmul
        # hazard).
        start = min(inside)
        if any(index(p) >= start for n in members for v in n.inputs
               if (p := graph.producer_of(v)) is not None and index(p) not in inside):
            continue
        # Exported values may also be read internally: fused_values must mean
        # *all* readers are internal, not merely that one reader is internal.
        fused = tuple(n.output for n in members if all(index(c) in inside for c in graph.consumers_of(n.output)))
        merged = FusionGroup(anchor,tuple(members),fused,
                             tuple(n for n in members if index(n)<index(anchor)),
                             tuple(n for n in members if index(n)>index(anchor)),
                             indexed=True,reduction_domain=domain,
                             reduction_batches=reduction_batches(graph,members,domain,coalesce=coalesce))
        if not group_results(graph,merged):
            continue
        groups = [g for i,g in enumerate(groups) if i not in selected]+[merged]
        groups=list(order_by_dependencies(graph, groups, index_of=index))
    return FusionReport(tuple(groups),frozenset(v for g in groups for v in g.fused_values))
