"""Operators the language defines by composition, expanded to executable nodes.

Some operators are not a kernel: they are an identity over operators the
compiler already lowers. `segmented_cumsum` is the first one --

    y[..., t] = cumsum(x)[..., t] - (cumsum(x)[..., s[t] - 1] if s[t] > 0 else 0)

-- and the identity needs no control flow and no new device code: a `cumsum`
(``flowgpu/compiler/backend/scan_backend.py``), a clamped index, and a last-axis gather
with a masked subtract (``flowgpu/compiler/backend/row_gather_backend.py``'s family).
Expanding it here, after semantic normalization and **before** fusion and the
13.7 buffer plan, is what makes that reuse real:

* the component nodes get their own kernel kinds, platform whitelist, precision
  contract, fusion placement and reference lowering -- exactly the treatment a
  program that wrote them by hand receives, so there is no second copy of the
  scan or of the gather to keep in step;
* the intermediate prefix is a normal value, so 13.7 materializes it with a
  normal slot. No backend and no buffer plan needs a new concept.

The expansion is the fail-closed boundary of the identity. `gather_rows`'
rule -- a negative index is normalised once by adding the extent, and one still
out of bounds reads zero -- would turn ``s[t] - 1`` at ``s[t] == 0`` into
``cumsum[..., -1]``, i.e. the row total, and be silently wrong. The expansion
therefore clamps the gather index with `maximum` to ``[0, N)`` and masks the
first-segment lanes to subtract zero, so no access depends on the gather
convention at all.

Two device passes fall out of this composition (scan, then gather + subtract).
The emitted kernel count is pinned by ``test/test_segmented_cumsum.py``; the
one-pass form -- folding the correction into a scan kernel's epilogue with a
register-level gather, or a single ``tl.cumsum``-plus-correction emitter -- is
the next step, written down in ``docs/OPERATOR_LOOP.md`` rather than implied.
"""
from flowgpu.compiler.core.diagnostic import CompileError, DiagnosticKind
from flowgpu.compiler.core.operators import COMPOSED_OPERATORS, OPERATORS
from flowgpu.compiler.ir.dataflow import DataflowGraph, Node, resolve_alias
from flowgpu.compiler.ir.semantic_ir import NormalizationDecision, OperationAttributes


def expand(graph):
    """Return ``(graph, decisions)`` with every composed operator expanded.

    The graph is rebuilt, not mutated: the input graph is the normalized
    dataflow that the semantic program produced, and keeping it intact is what
    lets a caller compare intent with executable form. A graph without a
    composed operator is returned unchanged.
    """
    if not any(node.op in COMPOSED_OPERATORS for node in graph.nodes):
        return graph, ()

    values = dict(graph.values)
    aliases = dict(graph.aliases)
    nodes = []
    decisions = []
    temporary = 0

    def resolve(name):
        return resolve_alias(aliases, name)

    def fresh():
        nonlocal temporary
        while True:
            name = f"%composed{temporary}"
            temporary += 1
            if name not in values:
                return name

    def type_of(op, inputs):
        return OPERATORS[op].infer([values[value] for value in inputs])

    def emit(op, inputs, output, part=None):
        nodes.append(Node(op, [resolve(value) for value in inputs], output, part,
                          OperationAttributes()))
        return output

    def stage(op, inputs):
        output = fresh()
        values[output] = type_of(op, inputs)
        return emit(op, inputs, output)

    for node in graph.nodes:
        if node.op not in COMPOSED_OPERATORS:
            nodes.append(Node(node.op, [resolve(value) for value in node.inputs],
                              node.output, node.part, node.attributes))
            continue
        if node.op != "segmented_cumsum":
            raise CompileError(
                DiagnosticKind.INTERNAL,
                f"composed operator '{node.op}' has no expansion",
                pipeline=graph.pipeline,
            )
        x, seg_start = (resolve(value) for value in node.inputs)

        # The scan keeps its own kernel: fp32 accumulation, one boundary round.
        prefix = stage("cumsum", [x])
        # The identity's subtracted index is `seg_start - 1`; clamp it so the
        # gather is proven in range and never reads the wrapped last element.
        one = stage("ones_like", [seg_start])
        shifted = stage("sub", [seg_start, one])
        # Separate zero tensors, not one shared value: fusion merges a chain
        # only while every intermediate has a single consumer, and sharing one
        # zeros_like between the clamp and the select would split the epilogue
        # into three extra launches (4 kernels vs 7 for the same arithmetic).
        zero_clamp = stage("zeros_like", [shifted])
        index = stage("maximum", [shifted, zero_clamp])
        gathered = stage("gather_last", [prefix, index])
        # A segment that starts at 0 has nothing before it: subtract zero.
        zero_mask = stage("zeros_like", [seg_start])
        prefix_zero = stage("zeros_like", [prefix])
        at_origin = stage("eq", [seg_start, zero_mask])
        subtractor = stage("where", [at_origin, prefix_zero, gathered])
        expected = type_of("sub", [prefix, subtractor])
        if values[node.output] != expected:
            raise CompileError(
                DiagnosticKind.INTERNAL,
                "segmented_cumsum expansion changed the declared result type: "
                f"{values[node.output]} and {expected}",
                pipeline=graph.pipeline,
            )
        emit("sub", [prefix, subtractor], node.output)
        decisions.append(NormalizationDecision(
            node.output, "expand_composition",
            "segmented_cumsum -> cumsum + clamped index + masked gather_last + sub; "
            "two device passes reuse the existing scan and gather lowerings",
        ))

    return DataflowGraph(graph.pipeline, values, nodes, aliases), tuple(decisions)
