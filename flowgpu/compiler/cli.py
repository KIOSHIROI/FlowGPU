"""FlowGPU command line interface.

    python main.py <path.fgpu> [--emit summary|ast|symbols|dataflow|fusion|buffers|tokens|json]

Exit codes: 0 compiled, 1 compilation error, 2 usage or I/O error.
Diagnostics go to stderr, results to stdout, so `--emit json` stays
machine-readable for a future generate/compile/repair loop.

The modes answer different questions about the same program: `summary` says
what it computes, `symbols` what each name is, `dataflow` what the Dataflow IR
(Design.md 7) looks like -- nodes, edges, and the value table that includes
the temporaries lowering introduced -- `fusion` which operators one kernel
evaluates (13.6), and `buffers` which values need storage, for how long, and
which of them share it (13.7, 11).
"""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import platform
import sys

from lark.exceptions import LarkError

from flowgpu.compiler.frontend.ast_node import (
    Assignment,
    Call,
    DTypeLiteral,
    InputDecl,
    Name,
    OutputDecl,
    ShapeLiteral,
    IntLiteral,
)
from flowgpu.compiler.compile import (
    DEFAULT_CROSS_DOMAIN_STRATEGY,
    CompileError,
    compile_source,
    syntax_error_from,
)
from flowgpu.compiler import __version__
from flowgpu.compiler.frontend.lexer import tokenize
from flowgpu.compiler.backend.reference import inputs_for, run as run_reference
from flowgpu.compiler.analysis.traffic import plan_traffic
from flowgpu.compiler.analysis.stability import PATTERNS, boundary, check
from flowgpu.compiler.core.operators import OPERATORS
from flowgpu.compiler.analysis.precision import ACC_FLOOR, declared, declined, width
from flowgpu.compiler.analysis.portability import measure as measure_portability
from flowgpu.compiler.backend.reference import module as reference_module
from flowgpu.compiler.backend.triton_backend import module
from flowgpu.compiler.backend.platform_configs import MAPPING_STRATEGIES, TRANSPOSE_MAPPINGS, PLATFORMS, resolve_platform

EMIT_CHOICES = (
    "summary",
    "ast",
    "symbols",
    "dataflow",
    "semantic",
    "fusion",
    "buffers",
    "stability",
    "precision",
    "portability",
    "contracts",
    "triton",
    "reference",
    "tokens",
    "json",
    "strategies",
    "gather-strategies",
)


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _format_type(tensor_type) -> str:
    # TensorType knows how the language writes it, so that a diagnostic and
    # the summary cannot disagree about the spelling of a type.
    return str(tensor_type)


def _format_expr(expression) -> str:
    if isinstance(expression, IntLiteral):
        return str(expression.value)
    if isinstance(expression, ShapeLiteral):
        return '[' + ', '.join(map(str, expression.shape)) + ']'
    if isinstance(expression, DTypeLiteral):
        # cast_to's second argument: a dtype, not a tensor (the default
        # --emit summary and --emit json crashed on every cast_to program
        # before this branch existed; review 2026-10-26)
        return expression.dtype
    if isinstance(expression, Name):
        return expression.name

    if isinstance(expression, Call):
        arguments = ", ".join(
            _format_expr(argument) for argument in expression.args
        )
        return f"{expression.name}({arguments})"

    # Loud on purpose: an unhandled node means this renderer has fallen behind
    # the AST, and printing "?" would hide it in the compiler's own output.
    raise AssertionError(
        f"cannot format expression of type {type(expression).__name__}"
    )


def _statements_of(pipeline_result, kind):
    return [
        statement
        for statement in pipeline_result.pipeline.statements
        if isinstance(statement, kind)
    ]


def _render_summary(pipeline_result) -> str:
    lines = [f"pipeline {pipeline_result.pipeline.name}"]

    inputs = _statements_of(pipeline_result, InputDecl)
    if inputs:
        lines.append("  inputs")
        for statement in inputs:
            lines.append(
                f"    {statement.name:<10} {_format_type(statement.type)}"
            )

    values = _statements_of(pipeline_result, Assignment)
    if values:
        lines.append("  values")
        for statement in values:
            rendered = _format_expr(statement.value)
            for number, name in enumerate(statement.names):
                # a multi-result operator binds one name per result, in the
                # operator's output order; the expression is printed once
                inferred = pipeline_result.symbols.lookup(name).type
                shown = rendered if number == 0 else "(continues)"
                lines.append(
                    f"    {name:<10} "
                    f"{_format_type(inferred):<14} "
                    f"= {shown}"
                )

    outputs = _statements_of(pipeline_result, OutputDecl)
    if outputs:
        lines.append("  outputs")
        for statement in outputs:
            inferred = pipeline_result.symbols.lookup(statement.name).type
            lines.append(
                f"    {statement.name:<10} {_format_type(inferred)}"
            )

    # The dependency comes from the Dataflow IR (13.5), not from a reading of
    # the tree: what a value depends on is a fact about the graph.
    edges = pipeline_result.dataflow.value_edges()
    if edges:
        lines.append("  dataflow")
        for producer, consumer in edges:
            lines.append(f"    {producer} -> {consumer}")

    return "\n".join(lines)


def _render_dataflow(pipeline_result) -> str:
    """The Dataflow IR: the value table, the nodes, and the edges."""
    graph = pipeline_result.dataflow
    lines = [f"pipeline {graph.pipeline}"]

    lines.append("  values")
    for name, tensor_type in graph.values.items():
        lines.append(f"    {name:<12} {_format_type(tensor_type)}")

    if graph.aliases:
        lines.append("  aliases")
        for name, target in graph.aliases.items():
            lines.append(f"    {name:<12} = {target}")

    lines.append("  nodes")
    width = len(str(len(graph.nodes) - 1)) if graph.nodes else 1

    for index, node in enumerate(graph.nodes):
        # An input node has no operands, so it reads better without the
        # empty parentheses: `input -> x`.
        signature = (
            f"{node.op}({', '.join(node.inputs)})" if node.inputs else node.op
        )
        label = f"n{index:<{width}}"

        if node.output is None:
            lines.append(f"    {label} {signature}")
        else:
            lines.append(f"    {label} {signature:<26} -> {node.output}")

    lines.append("  edges")
    for edge in graph.edges():
        lines.append(
            f"    n{graph.index_of(edge.producer)} -> "
            f"n{graph.index_of(edge.consumer)}   {edge.value}"
        )

    return "\n".join(lines)


def _render_fusion(pipeline_result) -> str:
    """13.6: which operators one kernel evaluates, and what that removes."""
    report = pipeline_result.fusion
    index_of = pipeline_result.dataflow.index_of

    lines = [f"pipeline {pipeline_result.dataflow.pipeline}"]
    lines.append("  kernels")

    width = len(str(len(report.groups) - 1)) if report.groups else 1

    for number, group in enumerate(report.groups):
        members = " -> ".join(
            f"n{index_of(node)} {node.op}" for node in group.members
        )
        lines.append(f"    k{number:<{width}} {members}")

        for value in group.fused_values:
            lines.append(f"    {'':<{width + 1}} fused  {value}")

    operators = sum(len(group.members) for group in report.groups)

    if report.fused_values:
        names = ", ".join(sorted(report.fused_values))
        outcome = (
            f"{_plural(len(report.fusions), 'fusion')}, "
            f"{_plural(len(report.fused_values), 'value')} kept in "
            f"registers: {names}"
        )
    else:
        outcome = "no fusion"

    lines.append(
        f"  {_plural(len(report.groups), 'kernel')} from "
        f"{_plural(operators, 'operator')}, {outcome}"
    )

    return "\n".join(lines)


def _traffic_as_json(pipeline_result, *, tiled=False) -> dict:
    """21's 显存流量, for a caller that reads the report rather than looks at it."""
    from flowgpu.compiler.backend.autotune import configs_for
    candidates = configs_for(pipeline_result.triton.platform_config.name) if pipeline_result.triton.autotune else ()
    bm = min(c[0] for c in candidates) if candidates else 64
    bn = min(c[1] for c in candidates) if candidates else 64
    traffic = plan_traffic(
        pipeline_result.dataflow, pipeline_result.fusion, pipeline_result.memory,
        tiled=tiled, block_m=bm, block_n=bn
    )

    return {
        "floor": traffic.floor_expression,
        "moved": traffic.moved_expression,
        "A": round(traffic.a, 4),
        "A_at": "every dimension = 1",
        "charges": [
            {
                "value": charge.value,
                "size": charge.size,
                "count": charge.count,
                "expression": charge.expression,
                "reason": charge.reason,
            }
            for charge in traffic.charges
        ],
    }


def _stability_as_json(pipeline_result) -> dict:
    """21.7's candidates, together with the boundary that keeps them empty."""
    return {
        "candidates": [
            {
                "primitive": candidate.primitive,
                "witness": list(candidate.witness),
                "values": list(candidate.values),
                "message": candidate.message(),
            }
            for candidate in check(pipeline_result.dataflow)
        ],
        "patterns": [
            {"primitive": pattern.primitive, "steps": list(pattern.steps)}
            for pattern in PATTERNS
        ],
        "boundary": list(boundary()),
    }


def _acc_of(graph, value) -> str | None:
    """21.6's `acc(v)`: what the operator that produced v accumulates in."""
    producer = graph.producer_of(value)

    if producer is None or producer.op not in OPERATORS:
        return None

    return OPERATORS[producer.op].acc


def _precision_as_json(pipeline_result) -> dict:
    """21.6's contract, and the two numbers the dimension is measured by."""
    graph = pipeline_result.dataflow

    return {
        "floor": ACC_FLOOR,
        "values": [
            {"value": value, "store": tensor_type.dtype, "acc": _acc_of(graph, value)}
            for value, tensor_type in graph.values.items()
        ],
        "declared": {name: OPERATORS[name].acc for name in declared(OPERATORS)},
        "declined": list(declined(OPERATORS)),
    }


def _portability_as_json(pipeline_result) -> dict:
    """21.8's numbers: the shared plan, and what each target reaches."""
    report = measure_portability(pipeline_result)

    return {
        "plan": {
            "kernels": report.kernels,
            "slots": report.slots,
            "A": round(report.a, 4),
            "computed": "once, from the plan 13.6 and 13.7 produce before any target",
        },
        "targets": [
            {
                "target": reach.target,
                "kernels": reach.kernels,
                "skipped": reach.skipped,
                "unreachable": list(reach.unreachable),
                "reasons": [list(pair) for pair in reach.reasons],
            }
            for reach in report.reaches
        ],
        "portable": report.portable,
        "unreachable_in_every_target": list(report.unreachable),
        "target_limits": list(report.target_limits),
    }


def _render_portability(pipeline_result) -> str:
    """21.8: one source, N targets, unchanged, and A not worse.

    A is printed once rather than per target, and that is the whole structural
    argument: 13.6 and 13.7 run before a target exists, so the two targets
    cannot disagree about the plan unless one of them ignores it.
    """
    report = measure_portability(pipeline_result)

    lines = [f"pipeline {pipeline_result.dataflow.pipeline}"]
    lines.append("  portability")
    lines.append(
        f"    plan        {report.kernels} kernels, {report.slots} slots, "
        f"A {report.a:.2f}   computed once, before any target"
    )

    for reach in report.reaches:
        summary = f"{reach.kernels} kernels"

        if reach.skipped:
            summary += f", {reach.skipped} skipped as dead"

        if reach.unreachable:
            summary += f", unreachable: {', '.join(reach.unreachable)}"
        else:
            summary += ", nothing unreachable"

        lines.append(f"    {reach.target:<11} {summary}")

    if report.portable:
        lines.append("    verdict     portable: one source, both targets")
    elif report.unreachable:
        lines.append(
            f"    verdict     not portable: {', '.join(report.unreachable)} "
            f"reaches neither target, which is a hole in the language rather "
            f"than a limit of a target"
        )
    else:
        lines.append(
            f"    verdict     not portable: {', '.join(report.target_limits)} "
            f"reaches some targets and not others"
        )

    return "\n".join(lines)


def _render_precision(pipeline_result) -> str:
    """21.6, per value and per contract.

    Rule 2 is a compile error rather than a warning, so there is no "violations"
    line to print: a program either satisfies the floor or it did not compile.
    What is left to report is the part that is a *declaration* -- who
    accumulates, in what, and who says it accumulates in nothing.
    """
    graph = pipeline_result.dataflow

    lines = [f"pipeline {graph.pipeline}"]
    lines.append("  precision   store is the value's dtype, acc is what its producer sums in")

    for value, tensor_type in graph.values.items():
        acc = _acc_of(graph, value) or "-"
        lines.append(f"    {value:<12} store {tensor_type.dtype:<6} acc {acc}")

    lines.append("  contract")
    for name in declared(OPERATORS):
        lines.append(f"    accumulate   {name:<12} {OPERATORS[name].acc}")

    for name in declined(OPERATORS):
        lines.append(f"    declined     {name:<12} accumulates nothing")

    narrowest = min(
        (width(OPERATORS[name].acc) for name in declared(OPERATORS)), default=0
    )
    lines.append(
        f"    {len(declared(OPERATORS))} reductions, none narrower than "
        f"{ACC_FLOOR}: every float accumulation is at least "
        f"{narrowest * 8} bits"
    )

    return "\n".join(lines)


def _render_contracts(pipeline_result) -> str:
    """The two backend contracts, side by side: the plan's, and the source's.

    They answer different questions about the same kernel. `analysis/access_shape`
    judges the *addressing* from the IR, before any source exists, and can only
    speak about a plan; `backend/contracts.violations` judges the *expressions* in
    the emitted text, and can only speak about code that has been written. A
    disagreement between them is the interesting case -- one of the two is wrong,
    and the earlier corrections in `research/kunlun_feasibility/RESULTS.md` were
    found exactly that way.

    This view exists so the comparison is a command rather than an exercise:
    printing only one of them invites reading a clean half as a clean whole.
    """
    from flowgpu.compiler.analysis.access_shape import group_shape
    from flowgpu.compiler.backend.contracts import violations
    from flowgpu.compiler.backend.platform_configs import resolve_platform

    platform = (pipeline_result.triton.platform_config
                if pipeline_result.triton is not None else None) or resolve_platform()
    triton = pipeline_result.triton
    kernels = list(triton.kernels) if triton is not None else []

    lines = [f"pipeline {pipeline_result.dataflow.pipeline}"]
    lines.append(f"  platform    {platform.name}")

    lines.append("  plan (from the IR, before emission)")
    shape = group_shape(pipeline_result.fusion)
    lines.append(f"    access      {shape.kind}  ({shape.reason})")
    for conflict in pipeline_result.platform_conflicts:
        lines.append(f"    conflict    {conflict}")
    if not pipeline_result.platform_conflicts:
        lines.append("    conflict    none")

    lines.append("  source (from the emitted kernels)")
    if not kernels:
        lines.append("    kernels     none emitted")
    total = 0
    for kernel in kernels:
        problems = violations(kernel.source, platform)
        total += len(problems)
        lines.append(f"    {kernel.name:<12} {'clean' if not problems else str(len(problems)) + ' violation(s)'}")
        for problem in problems:
            lines.append(f"      {problem}")
    if kernels and not total:
        lines.append("    violations  none")

    lines.append("  agreement")
    # A refused kernel is not in `kernels`, so the text half cannot judge it.
    # That is *partial*, not contradictory -- and saying "DISAGREE" there would
    # train the reader to ignore the one line that must not be ignored.
    planned = len(pipeline_result.fusion.groups)
    emitted = len(kernels)
    if not kernels:
        lines.append("    undetermined  nothing was emitted to compare against")
    elif bool(pipeline_result.platform_conflicts) == bool(total):
        lines.append("    agree         both halves say the same thing")
    elif pipeline_result.platform_conflicts and emitted < planned:
        lines.append(
            f"    partial       the plan refuses {planned - emitted} of {planned} "
            f"groups, so the text half had nothing to judge for them"
        )
    else:
        lines.append(
            "    DISAGREE      the plan and the source disagree about this "
            "target; one of the two checks is wrong"
        )
    return "\n".join(lines)


def _render_stability(pipeline_result) -> str:
    """21.7: did this pipeline spell out a primitive it could have called?

    The empty answer is the expected one, so the report says what would change
    it. A view that only ever printed "none" would be indistinguishable from a
    view whose check was broken.
    """
    lines = [f"pipeline {pipeline_result.dataflow.pipeline}"]
    lines.append("  stability")

    candidates = check(pipeline_result.dataflow)

    if candidates:
        for candidate in candidates:
            lines.append(f"    candidate   {candidate.message()}")
    else:
        lines.append("    candidates  none")

    lines.append("    patterns")
    for pattern in PATTERNS:
        lines.append(f"      {pattern.primitive:<12} {pattern.expansion}")

    missing = boundary()
    lines.append("    boundary")
    if missing:
        lines.append(
            f"      19 registers no {', '.join(missing)}, so no program can "
            f"express one of these yet"
        )
    else:
        lines.append(
            "      every operator these patterns need is registered, so this "
            "check can fire"
        )

    return "\n".join(lines)


def _render_buffers(pipeline_result) -> str:
    """11's Buffer metadata: size, lifetime, storage, for every value."""
    memory = pipeline_result.memory

    lines = [f"pipeline {pipeline_result.dataflow.pipeline}"]
    lines.append("  buffers")

    for buffer in memory.buffers:
        span = f"n{buffer.start}..n{buffer.end}"
        storage = buffer.storage if buffer.storage is not None else "-"

        # The size is padded only when a note follows it, so that the notes
        # line up without every other row ending in blanks.
        size = buffer.size if not buffer.note else f"{buffer.size:<10}"

        row = (
            f"    {buffer.value:<12} "
            f"{_format_type(buffer.tensor_type):<14} "
            f"{span:<12} "
            f"{storage:<8} "
            f"size {size}"
        )

        if buffer.note:
            row += f"   {buffer.note}"

        lines.append(row)

    if memory.slots:
        lines.append("  slots")
        for slot in memory.slots:
            lines.append(
                f"    slot{slot.index:<4} n{slot.start}..n{slot.end}   "
                f"{', '.join(slot.values)}"
            )

    allocated = len(memory.allocated)
    used = len(memory.slots)
    reused = allocated - used

    outcome = (
        f"{_plural(allocated, 'buffer')} allocated in "
        f"{_plural(used, 'slot')}"
    )
    if reused:
        outcome += f", {reused} reused"

    eliminated = len(memory.eliminated)
    if eliminated:
        outcome += f"; {_plural(eliminated, 'value')} not materialized"

    lines.append(f"  {outcome}")

    # 21's 显存流量, as a number rather than a definition. See flowgpu/compiler/traffic.py
    # for what the floor is and why attribution sits next to it.
    traffic = plan_traffic(
        pipeline_result.dataflow, pipeline_result.fusion, pipeline_result.memory
    )
    lines.append("  traffic (bytes, per element)")
    lines.append(f"    floor   {traffic.floor_expression}")
    lines.append(f"    moved   {traffic.moved_expression}")
    lines.append(f"    A       {traffic.a:.2f}   (every dimension = 1)")

    for charge in traffic.above_the_floor():
        lines.append(
            f"      {charge.value:<10} {charge.expression:<12} {charge.reason}"
        )

    return "\n".join(lines)


def _render_symbols(pipeline_result) -> str:
    lines = []
    for name, symbol in pipeline_result.symbols.symbols.items():
        inferred = (
            _format_type(symbol.type) if symbol.type is not None else "-"
        )
        lines.append(f"{symbol.kind:<6} {name:<12} {inferred}")
    return "\n".join(lines)


def _tensor_type_as_json(tensor_type) -> dict:
    # `shape: []` already says rank 0, but a reader of the JSON should not have
    # to know that an empty shape means "scalar" rather than "empty tensor".
    payload = {"dtype": tensor_type.dtype, "shape": list(tensor_type.shape)}
    if tensor_type.is_scalar:
        payload["scalar"] = True

    return payload


def _pipeline_as_json(pipeline_result) -> dict:
    def type_of(name):
        return _tensor_type_as_json(pipeline_result.symbols.lookup(name).type)

    graph = pipeline_result.dataflow
    index_of = graph.index_of

    fusion = pipeline_result.fusion
    memory = pipeline_result.memory

    return {
        "pipeline": pipeline_result.pipeline.name,
        "semantic": pipeline_result.semantic.as_dict(),
        "normalization": [dict(value=d.value, action=d.action, reason=d.reason)
                          for d in pipeline_result.normalization],
        "inputs": [
            {
                "name": statement.name,
                "type": _tensor_type_as_json(statement.type),
            }
            for statement in _statements_of(pipeline_result, InputDecl)
        ],
        "values": [
            {
                "name": name,
                "type": type_of(name),
                "expr": (_format_expr(statement.value) if number == 0
                         else "(continues)"),
            }
            for statement in _statements_of(pipeline_result, Assignment)
            for number, name in enumerate(statement.names)
        ],
        "outputs": [
            {"name": statement.name, "type": type_of(statement.name)}
            for statement in _statements_of(pipeline_result, OutputDecl)
        ],
        # The dependency as value -> value: what is computed from what.
        "edges": [
            {"from": producer, "to": consumer}
            for producer, consumer in graph.value_edges()
        ],
        # The Dataflow IR itself (Design.md 7): node-level edges, and the
        # value table, which is the only place a lowering temporary shows up.
        "dataflow": {
            "values": [
                {"name": name, "type": _tensor_type_as_json(tensor_type)}
                for name, tensor_type in graph.values.items()
            ],
            "aliases": dict(graph.aliases),
            "nodes": [
                {
                    "index": index,
                    "op": node.op,
                    "inputs": list(node.inputs),
                    "output": node.output,
                }
                for index, node in enumerate(graph.nodes)
            ],
            "edges": [
                {
                    "from": index_of(edge.producer),
                    "to": index_of(edge.consumer),
                    "value": edge.value,
                }
                for edge in graph.edges()
            ],
        },
        # 13.6: the operators each kernel evaluates. `fused_values` are the
        # ones that never need a buffer, which is the whole point of fusing.
        "fusion": {
            "kernels": [
                {
                    "index": number,
                    "anchor": group.anchor.op if group.anchor else None,
                    "indexed": group.indexed,
                    "row_gather": group.row_gather,
                    "row_gather_schedule": list(group.row_gather_schedule),
                    "reduction_domain": list(group.reduction_domain),
                    "reduction_batches": [list(batch) for batch in group.reduction_batches],
                    "cross_domain": list(group.cross_domain),
                    "cross_tile": list(group.cross_tile),
                    "members": [index_of(node) for node in group.members],
                    "fused_values": list(group.fused_values),
                    "prologue": [n.output for n in group.prologue],
                    "epilogue": [n.output for n in group.epilogue],
                }
                for number, group in enumerate(fusion.groups)
            ],
            "fused_values": sorted(fusion.fused_values),
        },
        # 13.7 and 11: what has to live in memory, for how long, and which
        # buffers share storage. `storage` is null for a value that needs
        # none, and "input" for one the caller provided.
        "platform_conflicts": list(pipeline_result.platform_conflicts),
        "stability": _stability_as_json(pipeline_result),
        "precision": _precision_as_json(pipeline_result),
        "portability": _portability_as_json(pipeline_result),
        "buffers": {
            "reduction_stages": asdict(memory.stage_plan) if memory.stage_plan is not None else None,
            # 21's 显存流量: T_floor, T_moved, A, and what each materialized
            # value costs. It sits with the buffers because it prices them.
            "traffic": _traffic_as_json(pipeline_result),
            "placement_traffic": _traffic_as_json(pipeline_result, tiled=True),
            "values": [
                {
                    "value": buffer.value,
                    "type": _tensor_type_as_json(buffer.tensor_type),
                    "start": buffer.start,
                    "end": buffer.end,
                    "size": buffer.size,
                    "storage": buffer.storage,
                    "note": buffer.note,
                }
                for buffer in memory.buffers
            ],
            "slots": [
                {
                    "index": slot.index,
                    "values": list(slot.values),
                    "start": slot.start,
                    "end": slot.end,
                }
                for slot in memory.slots
            ],
        },
        # 13.8: what the backend emitted, and what it could not. The generated
        # source itself is not repeated here -- `--emit triton` prints that --
        # but a refusal is a diagnostic, so a repair loop reads it from here.
        "triton": {
            "gpu_backend": pipeline_result.triton.gpu_backend,
            "attention_impl": pipeline_result.triton.attention_impl,
            "autotune_fold": pipeline_result.triton.autotune_fold,
            "kernel_abi_audit": list(pipeline_result.triton.kernel_abi_audit),
            "platform_config": pipeline_result.triton.platform_config.as_dict(),
            "complete": pipeline_result.triton.complete,
            "kernels": [
                {
                    "name": kernel.name,
                    "kind": kernel.kind,
                    "ops": list(kernel.ops),
                    "operands": list(kernel.operands),
                    "results": list(kernel.results),
                    "grid": kernel.grid,
                    "autotune_family": kernel.autotune_family,
                    "launch_options": dict(kernel.launch_options),
                }
                for kernel in pipeline_result.triton.kernels
            ],
            "refusals": [
                {
                    "operator": refusal.operator,
                    "values": list(refusal.values),
                    "reason": refusal.reason,
                }
                for refusal in pipeline_result.triton.refusals
            ],
        },
    }


def _as_json(result) -> dict:
    return {
        "source": result.source_path,
        "pipelines": [
            _pipeline_as_json(pipeline_result)
            for pipeline_result in result.pipelines
        ],
    }


def _read_source(path: str) -> str:
    if path == "-":
        return sys.stdin.read()

    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        prog="flowgpu",
        description=(
            "Compile a FlowGPU program: parse it, check it, "
            "and report the inferred types."
        ),
    )
    argument_parser.add_argument(
        "--version", action="version",
        version="FlowGPU " + __version__,
    )
    argument_parser.add_argument(
        "--doctor",
        action="store_true",
        help="print environment diagnostics and exit",
    )
    argument_parser.add_argument(
        "path",
        nargs="?",
        help="path to a .fgpu file, or - to read from stdin",
    )
    argument_parser.add_argument(
        "--emit",
        choices=EMIT_CHOICES,
        default=None,
        help="what to print on success (default: summary)",
    )
    argument_parser.add_argument(
        "--run",
        action="store_true",
        help="run the program with the reference backend and print its outputs",
    )
    argument_parser.add_argument(
        "--run-gpu",
        action="store_true",
        help="run the emitted Triton on a CUDA device and compare with the "
             "reference backend (needs torch + triton)",
    )
    argument_parser.add_argument(
        "--gpu-backend", choices=("triton", "hybrid"), default="triton",
        help="hybrid uses library GEMM/SDPA with Triton pointwise kernels",
    )
    argument_parser.add_argument(
        "--library-matmul", choices=("unfused", "all"), default="unfused",
        help="hybrid only: unfused routes bare GEMMs to cuBLAS while fused "
             "GEMM groups keep their Triton kernel (default); all sends "
             "every matmul to the library",
    )
    argument_parser.add_argument("--layout-on-device", action="store_true",
                                 help="emit device copy kernels for reshape/split/concat/interleave "
                                      "instead of host torch copies (default off; changes emitted bytes)")
    argument_parser.add_argument("--attention-impl", choices=("math", "sdpa"), default="math",
                                 help="attention implementation; math is default, sdpa is an experimental comparison")
    argument_parser.add_argument(
        "--platform-gate", action="store_true",
        help="refuse before emission when analysis/access_shape judges the plan's "
             "addressing unaffordable on the selected target (off by default: "
             "turning a checker into a refusal needs parent/candidate evidence)",
    )
    argument_parser.add_argument(
        "--custom-ops", action="store_true",
        help="append torch.library custom-op registrations to the emitted module",
    )
    def shape_binding(text):
        try:
            name, value = text.split("=", 1)
            value = int(value)
            if not name.isidentifier() or value <= 0:
                raise ValueError()
            return name, value
        except ValueError:
            raise argparse.ArgumentTypeError("shape must be NAME=positive_integer")
    argument_parser.add_argument("--shape", action="append", default=[], type=shape_binding,
                                 help="specialize a symbolic dimension; repeat for placement cost decisions and guards")
    argument_parser.add_argument("--autotune", action="store_true",
                                 help="autotune native GEMM with platform-pruned configs and persistent Triton timings")
    argument_parser.add_argument('--prune-kernel-abi', action='store_true',
        help='experimental: remove unused fixed Triton parameters and paired launch arguments')
    argument_parser.add_argument('--autotune-fold', action='store_true',
                                 help='opt-in CUDA rowwise/indexed-reduction tuning; independent from GEMM tuning')
    argument_parser.add_argument("--fusion-policy", choices=("rule", "traffic"), default="traffic",
                                 help="choose legal placements by estimated traffic (symbolic shapes keep conservative rules)")
    argument_parser.add_argument("--no-index-fusion", action="store_true",
                                 help="materialize index/broadcast/reduction intermediates for a same-source control")
    argument_parser.add_argument("--gather-fusion", action="store_true",
                                 help="opt-in CUDA shared-index sibling row gathers in one launch")
    argument_parser.add_argument('--no-view-folding', action='store_true',
                                 help='retain identity permutation chains for a same-source control')
    argument_parser.add_argument("--compare-partial-gathers", action="store_true", help="include explicit width-based partial gather partitions in comparison")
    argument_parser.add_argument("--emulate-fp32", action="store_true", help="scalar FP32 reference arithmetic; not a GPU instruction simulator")
    argument_parser.add_argument("--gather-plan", help="explicit JSON file of gather partitions and per-group mappings")
    argument_parser.add_argument("--gather-mapping", choices=('flat', 'row_tiled'), default='flat',
                                 help="row gather mapping; row_tiled uses --indexed-rows/--indexed-columns")
    argument_parser.add_argument("--no-reduction-fusion", action="store_true",
                                 help="keep the prior indexed partition without multi-output shared-domain sweeps")
    argument_parser.add_argument("--cache-dir", help="content-addressed emitted-module cache directory")
    argument_parser.add_argument("--allow-experimental-kernels", default="",
                                 help="explicit opt-in kernel kinds, comma-separated; enables source generation, not validation")
    argument_parser.add_argument("--platform", choices=tuple(PLATFORMS), default="cuda",
                                 help="pointwise platform profile; non-CUDA profiles are experimental")
    argument_parser.add_argument("--mapping-strategy", choices=MAPPING_STRATEGIES,
                                 help="pointwise tile scheduling; grid_stride_loop requires --device-cores")
    argument_parser.add_argument("--block-size", type=int, help="pointwise tile size, a positive power of two")
    argument_parser.add_argument('--linear-view-indexing', action='store_true',
                                 help='opt-in proven linear addressing for equal-spaced pointwise views')
    argument_parser.add_argument('--paired-view-loads', action='store_true',
                                 help='opt-in guarded paired reads of adjacent external pointwise views')
    argument_parser.add_argument('--broadcast-row-tiling', action='store_true',
                                 help='opt-in CUDA row-local tiles and scalar [M,1] broadcasts')
    argument_parser.add_argument("--device-cores", type=int, help="actual core count; Ascend uses Vector Cores")
    argument_parser.add_argument("--grid-multiplier", type=int, help="loop grid cap = device cores * multiplier")
    argument_parser.add_argument("--num-warps", type=int, help="pointwise / GEMM launch tuning parameter")
    argument_parser.add_argument("--num-stages", type=int, help="GEMM pipeline stages (not an autotuner)")
    argument_parser.add_argument("--group-m", type=int, help="GEMM grouped row tiles; 1 uses direct 2D mapping")
    argument_parser.add_argument("--transpose-mapping", choices=TRANSPOSE_MAPPINGS,
                                 help="tiled transpose or contiguous output gather; non-CUDA transpose remains experimental")
    argument_parser.add_argument("--reduction-block-limit", type=int,
                                 help="longest row one reduction tile holds; longer or symbolic rows lower as a "
                                      "chunked two-pass (kunlunxin default 128, from uni_sram evidence)")
    argument_parser.add_argument("--indexed-columns", type=int, help="explicit indexed R tile bound (power of two)")
    argument_parser.add_argument("--indexed-rows", type=int, help="explicit indexed output/D tile bound (power of two)")
    argument_parser.add_argument('--reduction-group-limit', type=int,
                                 help='opt-in number of reductions in one shared-domain region (default 2)')
    argument_parser.add_argument('--coalesce-reduction-sweeps', action='store_true',
                                 help='opt-in shared scans for independent reductions')
    argument_parser.add_argument('--experimental-region-fusion', action='store_true',
                                 help='opt-in indexed/reduction fusion on non-CUDA targets; requires device validation')
    argument_parser.add_argument("--cross-domain-strategy", choices=("auto", "materialize", "recompute", "segmented", "segmented-merge"), default=DEFAULT_CROSS_DOMAIN_STRATEGY,
                                 help="opt-in CUDA recomputation or bounded closed-region segmentation; materialize remains default")
    argument_parser.add_argument("--compare-recompute", action="store_true",
                                 help="include explicit recompute candidates in --emit strategies")
    argument_parser.add_argument("--compare-large-tiles", action="store_true",
                                 help="include budget32768/8-warp candidates; requires --compare-recompute")
    argument_parser.add_argument("--compare-segmented", action="store_true",
                                 help="include bounded staged reduction candidate; requires --compare-recompute")
    argument_parser.add_argument("--compare-merged-statistics", action="store_true",
                                 help="include two-stage shifted-exponential merge; requires --compare-segmented")
    argument_parser.add_argument("--strategy-evidence", help="exact-context resource/timing JSON for --emit strategies")
    argument_parser.add_argument("--strategy-policy", help="frozen experimental candidate-cost JSON for --emit strategies; reports a recommendation only")
    return argument_parser


def _render_symbols_for_all(result) -> str:
    multiple = len(result.pipelines) > 1

    blocks = []
    for pipeline_result in result.pipelines:
        body = _render_symbols(pipeline_result)
        if multiple:
            body = f"pipeline {pipeline_result.pipeline.name}\n{body}"
        blocks.append(body)

    return "\n\n".join(blocks)


def _report(error: CompileError, as_json: bool) -> None:
    """Write one diagnostic to stderr.

    A caller that asked for JSON gets JSON here too. Design.md 17.4's repair
    loop consumes the category and the position to decide what to fix and
    where; recovering them by parsing an English sentence back apart is not
    a diagnostic channel. stdout stays empty on failure in both forms, so
    success is still distinguishable by the stream alone.
    """
    diagnostics = error.all_diagnostics()

    if as_json:
        payload = error.as_dicts()
        # One diagnostic keeps the shape 17.4 already reads. Several arrive as
        # an array, because dropping them or silently keeping only the first
        # would make the loop repair one thing per compile again.
        print(
            json.dumps(payload[0] if len(payload) == 1 else payload),
            file=sys.stderr,
        )
        return

    for diagnostic in diagnostics:
        print(f"error: {diagnostic}", file=sys.stderr)


def _doctor() -> int:
    print("FlowGPU " + __version__)
    print("Python: " + platform.python_version() + " (" + platform.platform() + ")")
    print("Executable: " + sys.executable)
    from importlib.util import find_spec
    for package in ("lark", "torch", "triton"):
        if find_spec(package) is None:
            print(package + ": not installed")
            continue
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
        print(package + ": " + version)
    print("Reference backend: available")
    print("Doctor only reports the environment; it does not run a GPU kernel.")
    return 0


def main(argv=None) -> int:
    argument_parser = build_argument_parser()
    args = argument_parser.parse_args(argv)
    if args.doctor:
        return _doctor()
    if args.path is None:
        argument_parser.error("the following arguments are required: path")
        return 2
    if args.strategy_policy and args.emit != 'strategies':
        print('error: --strategy-policy requires --emit strategies', file=sys.stderr)
        return 2

    # `--run` is not a representation of the program, so it is a flag rather
    # than a mode of `--emit`; asking for both at once is a usage error rather
    # than something to resolve silently.
    if args.run and args.emit is not None:
        print("error: --run and --emit cannot be combined", file=sys.stderr)

        return 2

    if args.run_gpu and (args.run or args.emit is not None):
        print("error: --run-gpu runs and compares; do not combine it with "
              "--run or --emit", file=sys.stderr)
        return 2

    try:
        platform_config = resolve_platform(
            args.platform, mapping_strategy=args.mapping_strategy, block_size=args.block_size,
            device_cores=args.device_cores, grid_multiplier=args.grid_multiplier, num_warps=args.num_warps,
            num_stages=args.num_stages, group_m=args.group_m, transpose_mapping=args.transpose_mapping,
            reduction_block_limit=args.reduction_block_limit,
            indexed_columns=args.indexed_columns, indexed_rows=args.indexed_rows,
            gather_mapping=args.gather_mapping,
            reduction_group_limit=args.reduction_group_limit,
            coalesce_reduction_sweeps=args.coalesce_reduction_sweeps,
            experimental_region_fusion=args.experimental_region_fusion,
            linear_view_indexing=args.linear_view_indexing,
            paired_view_loads=args.paired_view_loads,
            broadcast_row_tiling=args.broadcast_row_tiling,
            allowed_kernel_kinds=(tuple(dict.fromkeys(PLATFORMS[args.platform].allowed_kernel_kinds +
                tuple(x.strip() for x in args.allow_experimental_kernels.split(",") if x.strip())))
                if args.allow_experimental_kernels else None),
            layout_on_device=True if args.layout_on_device else None,
        )
        if args.platform != "cuda" and args.gpu_backend != "triton":
            raise ValueError("non-CUDA platform profiles require --gpu-backend triton")
    except ValueError as exc:
        message = f"invalid platform mapping: {exc}"
        print(json.dumps({"kind": "configuration", "message": message}) if args.emit == "json"
              else f"error: {message}", file=sys.stderr)
        return 2

    if args.emit is None:
        args.emit = "summary"

    try:
        source = _read_source(args.path)
    except UnicodeDecodeError as exc:
        print(f"error: cannot read {args.path}: not valid UTF-8 ({exc})",
              file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot read {args.path}: {exc}", file=sys.stderr)
        return 2

    if args.emulate_fp32 and args.emit != 'reference' and not args.run:
        print('error: --emulate-fp32 requires --emit reference or --run', file=sys.stderr)
        return 2
    gather_plan = None
    if args.gather_plan:
        try:
            with open(args.gather_plan, encoding='utf-8') as handle:
                gather_plan = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 2

    if args.emit == "strategies":
        from flowgpu.compiler.optimization.strategy import compare_strategies
        try:
            if args.platform != "cuda" or any(v is not None for v in (
                args.indexed_columns, args.indexed_rows, args.reduction_block_limit,
                args.num_warps, args.num_stages, args.group_m, args.block_size,
                args.mapping_strategy, args.transpose_mapping, args.grid_multiplier,
            )) or args.linear_view_indexing or args.paired_view_loads or args.broadcast_row_tiling or args.cross_domain_strategy != DEFAULT_CROSS_DOMAIN_STRATEGY or args.gather_fusion or args.gather_plan or args.compare_partial_gathers or args.gather_mapping != 'flat':
                raise ValueError("strategy comparison owns CUDA candidate tiles; use --compare-recompute/--compare-large-tiles")
            evidence = None
            if args.strategy_evidence:
                with open(args.strategy_evidence, encoding="utf-8") as handle:
                    evidence = json.load(handle)
            report = compare_strategies(source, dict(args.shape), recompute=args.compare_recompute,
                large_tiles=args.compare_large_tiles, segmented=args.compare_segmented, merge_statistics=args.compare_merged_statistics, sm_count=args.device_cores, evidence=evidence,
                compile_options=dict(gpu_backend=args.gpu_backend, library_matmul=args.library_matmul,
                    attention_impl=args.attention_impl, custom_ops=args.custom_ops, autotune=args.autotune,
                    autotune_fold=args.autotune_fold, prune_kernel_abi=args.prune_kernel_abi,
                    fusion_policy=args.fusion_policy, index_fusion=not args.no_index_fusion,
                    reduction_fusion=not args.no_reduction_fusion,
                    semantic_view_folding=not args.no_view_folding))
            if args.strategy_policy:
                from flowgpu.compiler.optimization.strategy_cost import predict_candidate_policy
                with open(args.strategy_policy, encoding="utf-8") as handle:
                    policy = json.load(handle)
                report['experimental_recommendation'] = predict_candidate_policy(policy, report)
            print(json.dumps(report, indent=2))
            return 0
        except CompileError as exc:
            _report(exc, as_json=True)
            return 1
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.emit == 'gather-strategies':
        from flowgpu.compiler.optimization.gather_strategy import compare_gather_strategies
        try:
            if (args.platform != 'cuda' or args.gpu_backend != 'triton' or args.autotune or args.autotune_fold
                    or args.gather_fusion or args.gather_plan or args.gather_mapping != 'flat'
                    or args.no_index_fusion or args.no_reduction_fusion or args.no_view_folding or args.custom_ops
                    or args.prune_kernel_abi or args.linear_view_indexing or args.paired_view_loads or args.broadcast_row_tiling
                    or args.allow_experimental_kernels or args.compare_recompute or args.compare_large_tiles or args.compare_segmented or args.compare_merged_statistics
                    or args.cross_domain_strategy != DEFAULT_CROSS_DOMAIN_STRATEGY or args.strategy_policy
                    or any(v is not None for v in (args.block_size, args.indexed_columns, args.indexed_rows,
                        args.reduction_block_limit, args.num_warps, args.num_stages, args.group_m,
                        args.mapping_strategy, args.transpose_mapping, args.grid_multiplier))):
                raise ValueError('gather comparison owns native CUDA candidate mappings')
            evidence = None
            if args.strategy_evidence:
                with open(args.strategy_evidence, encoding='utf-8') as handle:
                    evidence = json.load(handle)
            print(json.dumps(compare_gather_strategies(source, dict(args.shape),
                             sm_count=args.device_cores, evidence=evidence, partial=args.compare_partial_gathers), indent=2))
            return 0
        except CompileError as exc:
            _report(exc, as_json=True)
            return 1
        except (ValueError, OSError) as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 2

    if args.cache_dir and args.emit == "triton" and not args.run:
        from flowgpu.compiler.runtime.cache import IncompleteModule, cached_module
        try:
            code, _, _ = cached_module(source, cache_dir=args.cache_dir,
                gpu_backend=args.gpu_backend, attention_impl=args.attention_impl,
                library_matmul=args.library_matmul,
                platform_config=platform_config, custom_ops=args.custom_ops,
                autotune=args.autotune, autotune_fold=args.autotune_fold, fusion_policy=args.fusion_policy, shape_bindings=dict(args.shape), index_fusion=not args.no_index_fusion, reduction_fusion=not args.no_reduction_fusion, cross_domain_strategy=args.cross_domain_strategy, gather_fusion=args.gather_fusion, gather_plan=gather_plan,
                semantic_view_folding=not args.no_view_folding,
            prune_kernel_abi=args.prune_kernel_abi, platform_gate=args.platform_gate)
            print(code, end="")
            return 0
        except IncompleteModule as exc:
            # a refusal is a compilation result: exit 1, exactly as the
            # uncached path reports it
            print(f"error: {exc}", file=sys.stderr)
            return 1
        except CompileError as exc:
            _report(exc, as_json=False)
            return 1
        except OSError as exc:
            print(f"error: cannot use cache directory {args.cache_dir}: {exc}",
                  file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    # The token stream is stage one on its own, so it is printed without
    # going through the parser.
    if args.emit == "tokens":
        try:
            tokens = list(tokenize(source))
        except LarkError as exc:
            _report(syntax_error_from(exc), as_json=args.emit == "json")
            return 1

        for token in tokens:
            print(f"{token.type:<10} {token.value}")
        return 0

    try:
        result = compile_source(source, source_path=args.path, gpu_backend=args.gpu_backend, library_matmul=args.library_matmul,
                                attention_impl=args.attention_impl, platform_config=platform_config,
                                custom_ops=args.custom_ops, autotune=args.autotune, autotune_fold=args.autotune_fold, fusion_policy=args.fusion_policy, shape_bindings=dict(args.shape), index_fusion=not args.no_index_fusion, reduction_fusion=not args.no_reduction_fusion, cross_domain_strategy=args.cross_domain_strategy, gather_fusion=args.gather_fusion, gather_plan=gather_plan,
                                semantic_view_folding=not args.no_view_folding, prune_kernel_abi=args.prune_kernel_abi, platform_gate=args.platform_gate)
    except CompileError as exc:
        _report(exc, as_json=args.emit == "json")
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.run:
        return _run(result, emulate_fp32=args.emulate_fp32)

    if args.run_gpu:
        return _run_gpu(result)

    if args.emit == "summary":
        print(
            "\n\n".join(
                _render_summary(pipeline_result)
                for pipeline_result in result.pipelines
            )
        )
        count = len(result.pipelines)
        suffix = "" if count == 1 else f" ({count} pipelines)"
        print(f"OK  {args.path}{suffix}")
    elif args.emit == "symbols":
        print(_render_symbols_for_all(result))
    elif args.emit == "dataflow":
        print(
            "\n\n".join(
                _render_dataflow(pipeline_result)
                for pipeline_result in result.pipelines
            )
        )
    elif args.emit == 'semantic':
        print(json.dumps([dict(**p.semantic.as_dict(),
                              normalization=[dict(value=d.value, action=d.action, reason=d.reason)
                                             for d in p.normalization])
                          for p in result.pipelines], indent=2))
    elif args.emit == "fusion":
        print(
            "\n\n".join(
                _render_fusion(pipeline_result)
                for pipeline_result in result.pipelines
            )
        )
    elif args.emit == "portability":
        print(
            "\n\n".join(
                _render_portability(pipeline_result)
                for pipeline_result in result.pipelines
            )
        )
    elif args.emit == "precision":
        print(
            "\n\n".join(
                _render_precision(pipeline_result)
                for pipeline_result in result.pipelines
            )
        )
    elif args.emit == "contracts":
        print(
            "\n\n".join(
                _render_contracts(pipeline_result)
                for pipeline_result in result.pipelines
            )
        )
    elif args.emit == "stability":
        print(
            "\n\n".join(
                _render_stability(pipeline_result)
                for pipeline_result in result.pipelines
            )
        )
    elif args.emit == "buffers":
        print(
            "\n\n".join(
                _render_buffers(pipeline_result)
                for pipeline_result in result.pipelines
            )
        )
    elif args.emit == "triton":
        # 13.8's artifact is a program, so it is all or nothing: a module
        # missing the kernels the backend refused is not a program, and
        # printing one that looked complete would be the plausible-but-wrong
        # output this compiler keeps refusing to produce.
        refused = [
            message
            for pipeline_result in result.pipelines
            for message in pipeline_result.triton.messages()
        ]

        if refused:
            for message in refused:
                print(f"error: {message}", file=sys.stderr)

            return 1

        # byte-identical with the cached path (`print(code, end="")`), which
        # used to differ by the trailing newline (review P2, 2026-10-26)
        print(module([pipeline_result.triton for pipeline_result in result.pipelines]),
              end="")
    elif args.emit == "reference":
        # The other target. Same rule as Triton: a module missing the kernels
        # the backend refused is not a program, and printing one that looked
        # complete would be the plausible-but-wrong output this compiler keeps
        # refusing to produce.
        refused = [
            message
            for pipeline_result in result.pipelines
            for message in pipeline_result.reference.messages()
        ]

        if refused:
            for message in refused:
                print(f"error: {message}", file=sys.stderr)

            return 1

        print(
            reference_module(
                [
                    pipeline_result.reference
                    for pipeline_result in result.pipelines
                ], emulate_fp32=args.emulate_fp32
            )
        )
    elif args.emit == "ast":
        for pipeline_result in result.pipelines:
            print(pipeline_result.pipeline)
    elif args.emit == "json":
        print(json.dumps(_as_json(result), indent=2))

    return 0


_GPU_DTYPE = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "fp64": "float64",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
}

#: output-dtype tolerance for comparing generated Triton with the fp64 reference
_GPU_TOLERANCE = {
    "fp16": (2e-2, 2e-2),
    "bf16": (2e-2, 2e-2),
    "fp32": (1e-4, 1e-4),
}


def _run_gpu(result) -> int:
    """Execute the emitted Triton modules on CUDA and compare with the reference.

    The compiler itself runs nothing.  This mode is the product quickstart: the
    reference backend computes the program in fp64 on fixed synthetic inputs,
    then the generated Triton runs those same numbers on the device and the
    outputs are compared per dtype.  It needs torch and triton; without a device
    it points the user at ``--run`` (reference backend, CPU only).
    """
    try:
        import torch  # noqa: F401 - optional dependency for this mode only
    except ImportError:
        print(
            "error: --run-gpu needs PyTorch; install torch+triton or use --run "
            "for the CPU reference backend",
            file=sys.stderr,
        )
        return 2
    if not torch.cuda.is_available():
        print(
            "error: --run-gpu found no CUDA device; use --run for the CPU "
            "reference backend",
            file=sys.stderr,
        )
        return 2

    from flowgpu.compiler.backend.triton_backend import module as triton_module

    torch_dtypes = {
        "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
        "fp64": torch.float64, "int32": torch.int32, "int64": torch.int64,
        "bool": torch.bool,
    }
    failures = 0
    for pipeline_result in result.pipelines:
        reference_program = pipeline_result.reference
        triton_program = pipeline_result.triton
        if not triton_program.complete:
            for message in triton_program.messages():
                print(f"error: {message}", file=sys.stderr)
            return 1
        sizes = {dimension: 2 for dimension in reference_program.dimensions}
        print(f"{reference_program.pipeline} (sizes {sizes}, CUDA)")
        try:
            expected_outputs = run_reference(reference_program, sizes)
        except Exception as error:  # noqa: BLE001 - generated reference host
            print(f"error: reference backend failed: {type(error).__name__}: {error}",
                  file=sys.stderr)
            return 2

        tensors = []
        for _name, shape, dtype, values in inputs_for(reference_program, sizes):
            if not shape:
                tensors.append(values)
                continue
            tensor = torch.tensor(values, dtype=torch_dtypes[dtype], device="cuda")
            tensors.append(tensor.reshape([int(dimension) for dimension in shape]))

        namespace = {}
        try:
            exec(compile(triton_module([triton_program]), "<gpu-run>", "exec"), namespace)
            host = namespace[f"run_{triton_program.pipeline}"]
            actual_outputs = host(*tensors)
        except Exception as error:  # noqa: BLE001 - generated device host
            print(f"error: generated Triton failed on the device: "
                  f"{type(error).__name__}: {error}", file=sys.stderr)
            return 2

        for name, tensor_type in reference_program.outputs:
            expected = expected_outputs[name]
            actual = actual_outputs[name]
            actual_values = actual.detach().reshape(-1).cpu().tolist()
            dtype = str(tensor_type.dtype)
            tolerance = _GPU_TOLERANCE.get(dtype)
            if tolerance is None:
                mismatched = sum(1 for got, want in zip(actual_values, expected)
                                 if int(got) != int(want))
                detail = f"{mismatched} mismatched elements"
                ok = mismatched == 0
            else:
                atol, rtol = tolerance
                worst = 0.0
                mismatched = 0
                for got, want in zip(actual_values, expected):
                    difference = abs(got - want)
                    allowed = atol + rtol * abs(want)
                    if difference == difference and difference > allowed:
                        mismatched += 1
                    worst = max(worst, difference if difference == difference else 0.0)
                detail = f"max abs diff {worst:.3g} ({mismatched} over tolerance)"
                ok = mismatched == 0
            print(f"  {name}  {tensor_type.dtype}{list(tensor_type.shape)}  "
                  f"{'PASS' if ok else 'FAIL'}  {detail}")
            failures += 0 if ok else 1
    if failures:
        print(f"error: {failures} output(s) differ from the reference", file=sys.stderr)
        return 1
    print("OK  generated Triton matches the reference backend on CUDA")
    return 0


def _run(result, *, emulate_fp32=False) -> int:
    """Run each pipeline with the reference backend and print its outputs.

    The compiler runs nothing itself. What runs here is the Python the
    reference backend just wrote (13.8), on inputs the CLI derives from the
    dimensions, so this is reproducible and needs no device -- which is the
    whole reason the reference backend exists. It says what the program
    computes; it says nothing about the generated Triton, which is still only
    read here.
    """
    refused = [
        message
        for pipeline_result in result.pipelines
        for message in pipeline_result.reference.messages()
    ]

    if refused:
        for message in refused:
            print(f"error: {message}", file=sys.stderr)

        return 1

    for pipeline_result in result.pipelines:
        program = pipeline_result.reference
        dimensions = ", ".join(
            f"{dimension}=2" for dimension in program.dimensions
        )
        print(f"{program.pipeline} ({dimensions})")

        try:
            values = run_reference(program, emulate_fp32=emulate_fp32)
        except Exception as error:  # noqa: BLE001 - the reference host is generated code
            # The reference host is generated Python: a lowering bug shows up as
            # an exception, and a traceback is not the CLI's exit-code contract
            # (review P1-11, 2026-10-26).  Say which pipeline produced it.
            print(
                f"error: pipeline '{program.pipeline}': running the reference "
                f"implementation failed: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            return 2

        for name, tensor_type in program.outputs:
            print(f"  {name}  {tensor_type}  {values[name]!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
