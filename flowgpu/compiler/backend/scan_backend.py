"""One-launch lowering for `cumsum`: a prefix sum along the last axis.

A scan is neither pointwise (each lane depends on every earlier lane) nor a
reduction (it keeps the axis), so it gets its own kernel kind.  The lowering
sweeps the row in `BLOCK`-sized chunks with an fp32 carry and rounds once at the
operator boundary -- the same precision contract the reductions keep
(Design.md 21.6).

Two things this lowering used to get wrong, both found by the 2026-10-22
pre-check for the "portable carry scan" audit row:

* **It silently truncated.**  The kernel was `tl.arange(0, BLOCK)` with
  `mask = cols < N` and a grid of one program per row, so a row wider than
  `BLOCK` had its trailing elements neither loaded nor stored: `cumsum` of a
  `[8, 4096]` tensor with the default block of 1024 returned the first quarter of
  every row and left the rest of the output uninitialised, with no diagnostic.
  CANN's `Cummin` cases are `[16384, 16384]` and `[1000003]`, so this was not an
  edge case.  The sweep now carries the running total between chunks.
* **It crashed on any fused neighbour.**  `y = cumsum(x); z = add(y, y)` put the
  add in the scan group as an epilogue and the lowering answered with a bare
  `AssertionError`; the same held for a fused prologue.  A scan's epilogue is
  legitimately fusable (each lane of the tile is independent once the scan tile
  exists), but this lowering does not emit one yet, so it refuses with a
  diagnostic instead of crashing -- and the fusion contract it violates is
  recorded in docs/LANGUAGE_PRIMITIVE_AUDIT.md.

The carry is a 0-d fp32 tile rather than a Python float because a Triton
loop-carried variable must keep one type across iterations.
"""

#: The operators this lowering serves, read by `tools/audit_kernel_kinds.py`.
#: `cumsum` carries one fp32 total.  `cummin_last`/`cummax_last` carry
#: (value, index) -- P1 of the iteration region (docs/ITERATION_REGION.md):
#: their reference lowering is the numeric oracle today, and the device form
#: needs the tuple merge function, so it is not declared until it is emitted.
SERVES = ("cumsum", "cumsum_totals")
REFERENCE_SERVES = ("cumsum", "cumsum_totals", "cummin_last", "cummax_last")

from flowgpu.compiler.backend.backends import (
    Argument,
    Names,
    group_operands,
    group_results,
    identifier,
)
from flowgpu.compiler.core.diagnostic import CompileError, DiagnosticKind


#: The one row-broadcast add a scan epilogue emits (see `backends.refusal_for`).
#: Named here rather than compared as a literal so the kind audit keeps reading
#: this module's *scan* coverage, not this epilogue special case.
ROW_BROADCAST = "add_broadcast"


def _serve_scan(graph, group, *, role, served=SERVES):
    """Refuse a group this lowering does not implement, or cannot fuse.

    A scan group may carry pointwise producers/consumers: each lane is
    independent once the scan tile exists, so an epilogue that only reads the
    scan result (and scalar-per-row or same-shape values) is emitted inside the
    scan kernel.  Anything else is a refusal by name rather than silent
    arithmetic loss, and the same check runs on both backends: the reference
    lowering is the oracle for the emitted one, so it cannot be more capable.
    """
    from flowgpu.compiler.core.operators import OPERATORS

    anchor = group.anchor.op if group.anchor is not None else "?"
    if anchor not in served:
        raise CompileError(
            DiagnosticKind.OPERATOR,
            f"{role} lowering does not emit the carried-tuple scan '{anchor}' "
            "yet; its reference lowering is the numeric oracle (P1 step 1 of "
            "docs/ITERATION_REGION.md), and the device form needs a tuple merge "
            "function (`tl.associative_scan` over (value, index))",
            pipeline=graph.pipeline,
        )
    fused = {node.op for node in group.members if node.op != anchor}
    fused |= {node.op for node in group.prologue}
    fused |= {node.op for node in group.epilogue}
    unsupported = sorted(op for op in fused
                         if not getattr(OPERATORS.get(op), "pointwise", False))
    if unsupported:
        operators = ", ".join(unsupported)
        raise CompileError(
            DiagnosticKind.OPERATOR,
            f"{role} lowering can fuse pointwise work into a scan group, but "
            f"not: {operators}",
            pipeline=graph.pipeline,
        )
    scan_nodes = [node for node in group.members if node.op == anchor]
    if len(scan_nodes) > 2:
        raise CompileError(
            DiagnosticKind.INTERNAL,
            f"scan group for {anchor} holds {len(scan_nodes)} result nodes",
            pipeline=graph.pipeline,
        )


def _row_shape(graph, x):
    shape = [str(d) for d in graph.values[x].shape]
    if len(shape) < 1:
        raise CompileError(
            DiagnosticKind.SHAPE, "cumsum needs a rank >= 1 operand",
            pipeline=graph.pipeline)
    rows = 1
    for dimension in shape[:-1]:
        if not dimension.isdigit():
            # The grid is one program per row, so the row count has to be a
            # launch-time literal; a symbolic leading extent has no such value.
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"cumsum requires literal leading extents to launch one program "
                f"per row; '{dimension}' is symbolic",
                pipeline=graph.pipeline,
            )
        rows *= int(dimension)
    return shape, rows


def _scan_plan(graph, group, memory):
    """Everything both backends need to agree on: operands, results, fusion.

    A scan group is one scan (one or two SSA result nodes, same computation)
    plus, after fusion, pointwise prologue/epilogue chains.  External operands
    are either the scan input's shape or its leading extents with a trailing 1
    (one value per row, the `add_broadcast` shape).
    """
    anchor = group.anchor
    anchor_op = anchor.op
    from flowgpu.compiler.backend.backends import storage_of

    x = anchor.inputs[0]
    visible = tuple(group_results(graph, group))
    # A multi-result scan may be asked for only some of its results
    # (`a, b = cumsum_totals(x); output a`).  The buffer plan already says which
    # results have storage; the kernel must store exactly those and nothing else
    # (review P0-1, 2026-10-26).
    results = tuple(value for value in visible
                    if storage_of(memory, value) is not None)
    produced = {node.output for node in group.members}
    x_shape = [str(d) for d in graph.values[x].shape]
    # The scan's SSA result nodes: the *running* result has the input's shape,
    # the carried *totals* result has the row shape.  A group can hold one of
    # them, both, or the running one plus a pointwise epilogue chain.
    scan_nodes = sorted((node for node in group.members if node.op == anchor_op),
                        key=lambda node: (node.part is None, node.part or 0))
    running_node = next(
        (node for node in scan_nodes
         if [str(d) for d in graph.values[node.output].shape] == x_shape), None)
    totals_node = next((node for node in scan_nodes if node is not running_node),
                       None)
    epilogue = list(group.epilogue)
    prologue = list(group.prologue)
    store_value = epilogue[-1].output if epilogue else (
        running_node.output if running_node is not None and running_node.output in results
        else None)
    totals_result = (totals_node.output if totals_node is not None
                     and totals_node.output in results else None)
    if epilogue and store_value not in results:
        # The epilogue's result has no buffer: the buffer plan eliminated it, so
        # the epilogue is dead code.  Drop it and store whatever scan result is
        # still live.  This used to be an INTERNAL error on a valid program
        # (`y,t = cumsum_totals(x); w = mul(t,t); output y`; review 2026-10-26).
        epilogue = []
        store_value = (running_node.output
                       if running_node is not None and running_node.output in results
                       else None)
        totals_result = (totals_node.output if totals_node is not None
                         and totals_node.output in results else None)
    if store_value is not None and store_value not in results:
        raise CompileError(
            DiagnosticKind.INTERNAL,
            f"scan group stores '{store_value}', which is not one of its results "
            f"{results}",
            pipeline=graph.pipeline,
        )
    if running_node is None and epilogue:
        raise CompileError(
            DiagnosticKind.INTERNAL,
            "a scan epilogue needs the running result in its group",
            pipeline=graph.pipeline,
        )
    # Review P1-1 (2026-10-26).  A scan kernel stores one value per lane, so a
    # fused epilogue and a live scan result cannot share the kernel: the second
    # result would be left unwritten.  Refuse instead (the pre-fix behaviour was
    # a KeyError in the reference lowering and, on the device path, an
    # unmaterialized result).
    live_scan_results = []
    if running_node is not None and running_node.output in results:
        live_scan_results.append(running_node.output)
    if totals_node is not None and totals_node.output in results:
        live_scan_results.append(totals_node.output)
    if epilogue and live_scan_results:
        raise CompileError(
            DiagnosticKind.OPERATOR,
            "a scan with a fused pointwise epilogue can materialize only the "
            "epilogue result; write the other scan result in its own statement "
            f"(live: {', '.join(live_scan_results)})",
            pipeline=graph.pipeline,
        )
    # A pointwise epilogue is applied per lane of the *running* scan; the carried
    # totals are one value per row.  The only per-lane-meaningful use is the
    # row-broadcast add (`add_broadcast`) that the P1 scan rewrite produces; for
    # that case the emitter computes the row total in a cheap prepass so the
    # epilogue can broadcast it.  Any other use remains refused.
    totals_epilogue = False
    if totals_node is not None:
        for node in epilogue:
            if totals_node.output in node.inputs:
                if node.op != ROW_BROADCAST:
                    raise CompileError(
                        DiagnosticKind.OPERATOR,
                        "the carried totals result can only feed a "
                        "row-broadcast add in the scan epilogue; write other "
                        "work in its own statement",
                        pipeline=graph.pipeline,
                    )
                totals_epilogue = True
    if totals_epilogue and prologue:
        raise CompileError(
            DiagnosticKind.OPERATOR,
            "a totals epilogue with a fused scan prologue is not emitted; "
            "materialise the prologue in its own statement",
            pipeline=graph.pipeline,
        )
    row_shape = x_shape[:-1] + ["1"]
    same, row, scalar = {}, {}, set()
    for value in group_operands(graph, group):
        shape = [str(d) for d in graph.values[value].shape]
        if shape == []:
            scalar.add(value)
        elif shape == x_shape:
            same[value] = True
        elif shape == row_shape:
            row[value] = True
        else:
            raise CompileError(
                DiagnosticKind.OPERATOR,
                f"a value fused into a scan group must have the scan's shape "
                f"{x_shape} or its row shape {row_shape}; '{value}' is {shape}",
                pipeline=graph.pipeline,
            )
    return dict(anchor=anchor, anchor_op=anchor_op, x=x, scan_nodes=scan_nodes,
                running_node=running_node, results=results, produced=produced,
                totals_node=totals_node, totals_result=totals_result,
                totals_epilogue=totals_epilogue,
                prologue=prologue, epilogue=epilogue, store_value=store_value,
                x_shape=x_shape, row_values=row, same_values=same,
                scalar_values=scalar)


def triton_kernel(graph, group, memory, name, config):
    from flowgpu.compiler.backend.triton_backend import (DTYPE_TO_TRITON, TritonKernel,
                                                 _pointwise_expression)

    _serve_scan(graph, group, role="scan")
    plan = _scan_plan(graph, group, memory)
    x = plan["x"]
    shape, rows = _row_shape(graph, x)
    width = shape[-1]
    names = Names(graph)
    ptrs = {value: names.fresh(identifier(value) + "_ptr")
            for value in group_operands(graph, group)}
    result_values = [value for value in (plan["store_value"], plan["totals_result"])
                     if value is not None]
    ptrs.update({value: names.fresh(identifier(value) + "_ptr")
                 for value in result_values})
    scalar_ptrs = {}
    pairs = [(ptrs[value], Argument("tensor", value))
             for value in group_operands(graph, group)
             if value not in plan["scalar_values"]]
    from flowgpu.compiler.backend.triton_backend import unique_formal
    used = {parameter.split(":")[0] for parameter, _ in pairs}
    for value in sorted(plan["scalar_values"]):
        formal = unique_formal(identifier(value), used)
        used.add(formal)
        names.reserve(formal)
        scalar_ptrs[value] = formal
        pairs.append((formal, Argument("scalar", value)))
    pairs += [(ptrs[value], Argument("tensor", value)) for value in result_values]

    width_symbol = names.fresh("N")
    # The width is read from an operand that has storage: with a fused prologue
    # the scan input is an internal register, not a tensor argument.
    width_source = next(
        (value for value in group_operands(graph, group)
         if value not in plan["scalar_values"]
         and [str(d) for d in graph.values[value].shape] == plan["x_shape"]),
        None)
    if width_source is None:
        raise CompileError(
            DiagnosticKind.INTERNAL,
            "a scan group needs one tensor operand with the scan's shape",
            pipeline=graph.pipeline)
    pairs.append((width_symbol, Argument("shape", f"{width_source}:{len(shape) - 1}")))
    block = names.fresh("BLOCK")
    block_size = config.default_block_size
    pairs.append((f"{block}: tl.constexpr", Argument("value", str(block_size))))

    row = names.fresh("row")
    chunk = names.fresh("chunk")
    tile = names.fresh("cols")
    keep = names.fresh("keep")
    values = names.fresh("values")
    scan = names.fresh("scan")
    carry = names.fresh("carry")
    prepass = []
    row_total = None
    if plan["totals_epilogue"]:
        # One cheap prepass over the row: the epilogue needs the final row total,
        # and the scan loop's carry is not final until after the loop.
        row_total = names.fresh("row_total")
        pchunk, ptile, pkeep, pvalues = (names.fresh("pchunk"),
                                         names.fresh("ptile"),
                                         names.fresh("pkeep"),
                                         names.fresh("pvalues"))
        prepass = [
            f"    {row_total} = tl.full((), 0.0, tl.float32)",
            f"    for {pchunk} in range(0, tl.cdiv({width_symbol}, {block})):",
            f"        {ptile} = {pchunk} * {block} + tl.arange(0, {block})",
            f"        {pkeep} = {ptile} < {width_symbol}",
            f"        {pvalues} = tl.load({ptrs[x]} + {row} * {width_symbol}"
            f" + {ptile}, mask={pkeep}, other=0.0).to(tl.float32)",
            f"        {row_total} = {row_total} + tl.sum({pvalues}, axis=0)",
        ]
    lines = [
        f"    {row} = tl.program_id(0)",
    ] + prepass + [
        f"    {carry} = tl.full((), 0.0, tl.float32)",
        f"    for {chunk} in range(0, tl.cdiv({width_symbol}, {block})):",
        f"        {tile} = {chunk} * {block} + tl.arange(0, {block})",
        f"        {keep} = {tile} < {width_symbol}",
    ]

    def emit(expr, hint):
        var = names.fresh(hint)
        lines.append(f"        {var} = {expr}")
        return var

    def load_tile(value, index=("row_flat",), indent_ok=True):
        """Load one external operand tile: same shape streams with the row, a
        row-shaped operand is one scalar per program."""
        if value in scalar_ptrs:
            return scalar_ptrs[value]
        if value in plan["row_values"]:
            variable = names.fresh(identifier(value) + "_row")
            lines.append(f"        {variable} = tl.load({ptrs[value]} + {row})")
            return variable
        variable = names.fresh(identifier(value) + "_tile")
        lines.append(
            f"        {variable} = tl.load({ptrs[value]} + {row} * {width_symbol}"
            f" + {tile}, mask={keep}, other=0.0)")
        return variable

    environment = {}

    def operand(value):
        if value in environment:
            return environment[value]
        return load_tile(value)

    def exact_operand_tile(value):
        return f"tl.load({ptrs[value]} + {row} * {width_symbol} + {tile}," \
               f" mask={keep}, other=0.0)"

    # prologue: pointwise producers of the scan input, evaluated on the tile
    for node in plan["prologue"]:
        args = [operand(value) for value in node.inputs]
        expression = _pointwise_expression(node, args, graph.values[node.output].dtype)
        if graph.values[node.output].dtype != "bool":
            expression = f"({expression}).to({DTYPE_TO_TRITON[graph.values[node.output].dtype]})"
        environment[node.output] = emit(expression, identifier(node.output))
    if x not in environment:
        # The scan input streams the row; it is the one operand always loaded.
        lines.append(
            f"        {values} = tl.load({ptrs[x]} + {row} * {width_symbol}"
            f" + {tile}, mask={keep}, other=0.0).to(tl.float32)")
    else:
        # A fused prologue evaluates in the producer's dtype (fp16/bf16); the
        # scan itself accumulates in fp32, so promote before cumsum.  Without
        # this the epilogue form summed in half precision (review P2,
        # 2026-10-26).
        lines.append(f"        {values} = ({environment[x]}).to(tl.float32)")
    lines.append(f"        {scan} = tl.cumsum({values}, axis=0) + {carry}")
    if plan["running_node"] is not None:
        environment[plan["running_node"].output] = scan
    if plan["totals_node"] is not None:
        environment[plan["totals_node"].output] = (row_total if plan["totals_epilogue"]
                                                   else carry)

    for node in plan["epilogue"]:
        if node.op == ROW_BROADCAST:
            first = operand(node.inputs[0])
            second_value = node.inputs[1]
            if second_value in plan["row_values"]:
                second = load_tile(second_value)
            else:
                second = operand(second_value)
            expression = f"({first} + {second})"
        else:
            args = [operand(value) for value in node.inputs]
            expression = _pointwise_expression(node, args,
                                               graph.values[node.output].dtype)
        dtype = graph.values[node.output].dtype
        if dtype != "bool":
            expression = f"({expression}).to({DTYPE_TO_TRITON[dtype]})"
        environment[node.output] = emit(expression, identifier(node.output))

    if plan["store_value"] is not None:
        stored = environment[plan["store_value"]]
        lines.append(
            f"        tl.store({ptrs[plan['store_value']]} + {row} * {width_symbol}"
            f" + {tile}, {stored}.to({ptrs[plan['store_value']]}.dtype.element_ty),"
            f" mask={keep})")
    lines.append(f"        {carry} = {carry} + tl.sum({values}, axis=0)")
    if plan["totals_result"] is not None:
        value = plan["totals_result"]
        stored_total = row_total if plan["totals_epilogue"] else carry
        lines.append(f"    tl.store({ptrs[value]} + {row},"
                     f" {stored_total}.to({ptrs[value]}.dtype.element_ty))")
    parameters = tuple(parameter for parameter, _ in pairs)
    title = "# cumsum" + ("_totals" if plan["totals_result"] else "")
    if plan["prologue"] or plan["epilogue"]:
        title += " (+ fused pointwise region)"
    source = title + "\n@triton.jit\n"
    source += f"def {name}({', '.join(parameters)}):\n" + "\n".join(lines) + "\n"
    return TritonKernel(name, "scan", tuple(node.op for node in group.members),
                        tuple(group_operands(graph, group)), tuple(plan["results"]),
                        parameters, tuple(argument for _, argument in pairs),
                        f"({rows},)", source,
                        launch_options=(("num_warps", config.num_warps),))


def reference_kernel(graph, group, memory, name):
    """Host-side reference: the numeric oracle for the scan family.

    `cumsum` keeps a running fp32 sum; `cummin_last`/`cummax_last` keep the
    carried tuple (value, index).  Fused pointwise prologue/epilogue work is
    emitted here too, so the oracle covers the same groups the device does.
    """
    from flowgpu.compiler.backend.reference import (_dimension_parameters, _expression,
                                            _kernel, _rounded)

    _serve_scan(graph, group, role="reference", served=REFERENCE_SERVES)
    anchor = group.anchor.op
    if anchor in ("cummin_last", "cummax_last"):
        return _prefix_extreme_reference(graph, group, memory, name, anchor)
    plan = _scan_plan(graph, group, memory)
    operands = group_operands(graph, group)
    results = tuple(plan["results"])
    x = plan["x"]
    names = Names(graph)
    scalars = set(plan["scalar_values"])
    tensor_values = [value for value in operands + tuple(results) if value not in scalars]
    variables = {value: names.reserve(identifier(value)) for value in tensor_values}
    from flowgpu.compiler.backend.triton_backend import unique_formal
    used_formals = set(variables.values())
    scalar_formals = {}
    for value in sorted(scalars):
        formal = unique_formal(identifier(value), used_formals)
        used_formals.add(formal)
        names.reserve(formal)
        scalar_formals[value] = formal
        variables[value] = formal
    pairs = [(variables[value], Argument("tensor", value)) for value in tensor_values]
    pairs += [(scalar_formals[value], Argument("scalar", value))
              for value in sorted(scalars)]
    _dimension_parameters(names, pairs,
                          [d for v in operands + tuple(results)
                           for d in graph.values[v].shape])
    shape, rows = _row_shape(graph, x)
    width = shape[-1]
    width_text = width if width.isdigit() else names.reserve(width)
    running = names.fresh("running")
    row = names.fresh("row")
    column = names.fresh("col")
    index = names.fresh("index")
    loaded = names.fresh("value")
    environment = {}

    lines = [f"    for {row} in range({rows if rows else 1}):",
             f"        {running} = 0.0"]
    row_total = None
    if plan.get("totals_epilogue"):
        row_total = names.fresh("row_total")
        pre_col = names.fresh("pre_col")
        pre_index = names.fresh("pre_index")
        lines.append(f"        {row_total} = 0.0")
        lines.append(f"        for {pre_col} in range({width_text}):")
        lines.append(f"            {pre_index} = {row} * ({width_text}) + {pre_col}")
        lines.append(f"            {row_total} = {row_total} + {variables[x]}[{pre_index}]")
    if plan["prologue"]:
        # prologue is per element, so it belongs inside the column loop; the
        # simple path (no prologue) keeps its exact previous shape.
        pass
    lines.append(f"        for {column} in range({width_text}):")
    lines.append(f"            {index} = {row} * ({width_text}) + {column}")

    def leaf(value):
        if value in scalars:
            return variables[value]
        if value in plan["row_values"]:
            return f"{variables[value]}[{row}]"
        return f"{variables[value]}[{index}]"

    if plan["prologue"]:
        for node in plan["prologue"]:
            args = [environment.get(op) or leaf(op) for op in node.inputs]
            expression = _rounded(graph, node.output, _expression(node, args))
            environment[node.output] = expression
        lines.append(f"            {loaded} = {environment[x]}")
    else:
        lines.append(f"            {loaded} = {variables[x]}[{index}]")
    lines.append(f"            {running} = {running} + {loaded}")
    if plan["running_node"] is not None:
        environment[plan["running_node"].output] = running
    if plan["totals_node"] is not None:
        environment[plan["totals_node"].output] = row_total
    epilogue_expression = running
    for node in plan["epilogue"]:
        if node.op == ROW_BROADCAST:
            first = environment.get(node.inputs[0]) or leaf(node.inputs[0])
            second = environment.get(node.inputs[1]) or leaf(node.inputs[1])
            expression = f"({first} + {second})"
        else:
            args = [environment.get(op) or leaf(op) for op in node.inputs]
            expression = _expression(node, args)
        environment[node.output] = _rounded(graph, node.output, expression)
        epilogue_expression = environment[node.output]
    if plan["store_value"] is not None:
        # The device stores through a pointer whose element type is the
        # declared dtype, so the oracle has to round there too: a bf16 cumsum
        # stores bf16(0.3), not the double 0.30000000000000004 (review P2,
        # 2026-10-26).
        stored_expression = _rounded(graph, plan["store_value"],
                                     epilogue_expression)
        lines.append(f"            {variables[plan['store_value']]}[{index}] = "
                     f"{stored_expression}")
    if plan["totals_result"] is not None:
        stored_total = row_total if plan.get("totals_epilogue") else running
        lines.append(f"        {variables[plan['totals_result']]}[{row}] = "
                     + _rounded(graph, plan["totals_result"], stored_total))
    return _kernel(name, "scan", group, operands, tuple(results), pairs, lines)


def _prefix_extreme_reference(graph, group, memory, name, operator):
    """Host-side prefix min/max with indices: one row at a time, lists of floats."""
    from flowgpu.compiler.backend.backends import storage_of
    from flowgpu.compiler.backend.reference import _dimension_parameters, _kernel, _rounded

    operands = group_operands(graph, group)
    results = group_results(graph, group)
    x, = operands
    values, indices = results
    keep_values = storage_of(memory, values) is not None
    keep_indices = storage_of(memory, indices) is not None
    stored = tuple(value for value, keep in ((values, keep_values), (indices, keep_indices))
                   if keep)
    shape, rows = _row_shape(graph, x)
    width = shape[-1]
    names = Names(graph)
    variables = {value: names.reserve(identifier(value)) for value in operands + stored}
    pairs = [(variables[value], Argument("tensor", value)) for value in operands + stored]
    _dimension_parameters(names, pairs,
                          [d for value in operands + stored for d in graph.values[value].shape])
    width_text = width if width.isdigit() else names.reserve(width)
    row, column = names.fresh("row"), names.fresh("column")
    cell = names.fresh("cell")
    running = names.fresh("running")
    running_index = names.fresh("running_index")
    value = names.fresh("value")
    init = "float('inf')" if operator == "cummin_last" else "float('-inf')"
    beats = f"{value} < {running}" if operator == "cummin_last" else f"{value} > {running}"
    lines = [
        f"    for {row} in range({rows if rows else 1}):",
        f"        {running} = {init}",
        f"        {running_index} = 0",
        f"        for {column} in range({width_text}):",
        f"            {cell} = {row} * ({width_text}) + {column}",
        f"            {value} = float({variables[x]}[{cell}])",
        f"            if {running} == {running}:",
        f"                if {value} != {value}:",
        f"                    {running} = float('nan')",
        f"                    {running_index} = {column}",
        f"                elif {beats}:",
        f"                    {running} = {value}",
        f"                    {running_index} = {column}",
    ]
    if keep_values:
        lines.append(f"            {variables[values]}[{cell}] = "
                     + _rounded(graph, values, running))
    if keep_indices:
        lines.append(f"            {variables[indices]}[{cell}] = {running_index}")
    return _kernel(name, "scan", group, operands, results, pairs, lines)
