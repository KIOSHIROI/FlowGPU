"""P1: fold a row-total reduction and a prefix scan into one scan + epilogue.

The measured prefix-scan pattern

    t = sum_axis(x, 1)
    c = cumsum_axis(x, 1)
    out = add(c, broadcast_like(t, x))

spends three kernels: a rowwise sum, the scan, and a pointwise broadcast add.
The scan lowering already carries the running state whose final value *is* the
row total (`cumsum_totals`), and the language has `add_broadcast` for exactly
the row-broadcast add.  This pass rewrites the pattern to

    c, t = cumsum_totals(x)
    out  = add_broadcast(c, t)

so the scan and its epilogue can share a kernel.  It is opt-in because it changes
emission and the backend support for `cumsum_totals` must be exercised by the
parent/candidate evidence.
"""
from __future__ import annotations

from flowgpu.compiler.core.ast import Assignment, Call, IntLiteral, Name, PipelineDecl


def _call(statement, names):
    if not isinstance(statement, Assignment) or len(statement.names) != 1:
        return None
    value = statement.value
    if isinstance(value, Call) and value.name in names:
        return value
    return None


def _name_arg(call, index):
    arg = call.args[index] if len(call.args) > index else None
    return arg.name if isinstance(arg, Name) else None


def _literal_arg(call, index):
    arg = call.args[index] if len(call.args) > index else None
    return arg.value if isinstance(arg, IntLiteral) else None


def _find_assignment(statements, target_name):
    for index, statement in enumerate(statements):
        if isinstance(statement, Assignment) and statement.names == [target_name]:
            return index, statement
    return None, None


def rewrite_pipeline(pipeline) -> PipelineDecl:
    statements = list(pipeline.statements)
    for add_index, add_statement in enumerate(statements):
        add_call = _call(add_statement, {"add", "add_broadcast"})
        if add_call is None or len(add_call.args) != 2:
            continue
        # one operand is the scan, the other an inline broadcast_like(total, x)
        scan_arg = broadcast_arg = None
        for arg in add_call.args:
            if isinstance(arg, Call) and arg.name == "broadcast_like" and len(arg.args) == 2:
                broadcast_arg = arg
            elif isinstance(arg, Name):
                scan_arg = arg.name
        if scan_arg is None or broadcast_arg is None:
            continue
        total_name = _name_arg(broadcast_arg, 0)
        source_name = _name_arg(broadcast_arg, 1)
        if total_name is None or source_name is None:
            continue
        sum_index, sum_statement = _find_assignment(statements, total_name)
        sum_call = _call(sum_statement, {"sum_axis", "sum_last"})
        if sum_call is None or len(sum_call.args) < 2:
            continue
        if _name_arg(sum_call, 0) != source_name:
            continue
        axis = _literal_arg(sum_call, 1)
        scan_index, scan_statement = _find_assignment(statements, scan_arg)
        scan_call = _call(scan_statement, {"cumsum_axis", "cumsum"})
        if scan_call is None or len(scan_call.args) < 2:
            continue
        if _name_arg(scan_call, 0) != source_name or _literal_arg(scan_call, 1) != axis:
            continue
        # The total must be used only by this add, and the scan only here and
        # by the add (single-use guarantees the rewrite preserves the graph).
        total_uses = sum(1 for statement in statements
                         if isinstance(statement, Assignment)
                         and _contains_name(statement.value, total_name))
        scan_uses = sum(1 for statement in statements
                        if isinstance(statement, Assignment)
                        and _contains_name(statement.value, scan_arg))
        if total_uses != 1 or scan_uses != 1:
            continue
        new_scan = Assignment([scan_arg, total_name],
                              Call("cumsum_totals", [Name(source_name)]))
        new_add = Assignment([add_statement.names[0]],
                             Call("add_broadcast", [Name(scan_arg), Name(total_name)]))
        rewritten = []
        for index, statement in enumerate(statements):
            if index == sum_index:
                continue
            if index == scan_index:
                rewritten.append(new_scan)
            elif index == add_index:
                rewritten.append(new_add)
            else:
                rewritten.append(statement)
        return PipelineDecl(pipeline.name, rewritten, position=pipeline.position)
    return pipeline


def _contains_name(value, name) -> bool:
    if isinstance(value, Name):
        return value.name == name
    if isinstance(value, Call):
        return any(_contains_name(arg, name) for arg in value.args)
    return False


def rewrite(pipelines):
    return [rewrite_pipeline(pipeline) for pipeline in pipelines]
