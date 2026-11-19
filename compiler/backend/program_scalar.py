"""Emission support for program-uniform scalars (IR: `ir/program_scalar_ir`).

One device load per program, folded into a scalar base. The functions here are
deliberately small and shared by every emitter that can host the value, because
the failure mode this feature guards against is a lowering that re-derives the
value *per lane*: the load would become a gather and the write address would
stop being `scalar_base + affine(arange)` -- the shape the measured targets
collapse on. So there is exactly one place that emits the load, one place that
folds a base, and one place that hands the host its bounds check.

Everything is source text, like the rest of the backends; nothing here runs
Triton. The IR contract (integer dtype, rank-0 index, fail-closed bounds) is
enforced by `compiler.ir.program_scalar_ir`; this module only decides how those
decisions reach the kernel and the host.
"""
from compiler.core.diagnostic import CompileError, DiagnosticKind


def _parameter_for(pairs, kind, value):
    """The formal name a kernel already carries for one IR value."""
    for parameter, argument in pairs:
        if argument.kind == kind and argument.text == value:
            return parameter
    return None


def index_source(spec, pairs):
    """The emitted scalar expression the table is indexed with.

    A program-id expression is uniform by definition; a literal is a
    compile-time constant; a host scalar is a plain kernel formal. A device
    scalar (`ScalarRef` with ``host_visible=False``) is only admitted by the IR
    under ``bounds="trusted"`` and is referenced like any other scalar formal.
    """
    index = spec.index
    if hasattr(index, "expression"):
        return index.expression
    if type(index) is int:
        return str(index)
    # ScalarRef: the caller's own name, which the emitter has bound either as a
    # host scalar formal or as another program-uniform scalar.
    parameter = _parameter_for(pairs, "scalar", index.name)
    if parameter is None:
        raise CompileError(
            DiagnosticKind.INTERNAL,
            f"program scalar '{spec.table}' is indexed by '{index.name}', but "
            f"that scalar is not a formal of the kernel that hosts it; the "
            f"emitter must bind a rank-0 value before the program scalar uses it",
        )
    return parameter


def emit_uniform_load(spec, table, hint, graph, names, pairs, layouts, lines,
                      *, grid_count, index_source_override=None):
    """Emit ``hint = tl.load(table_ptr + index)`` once and return its binding.

    Returns ``(variable, bound)`` where ``bound`` is the host-side bounds
    record. The load is emitted at the kernel's top level (not inside a lane
    expression), so the device performs one scalar load per program. Shared by
    every lowering that reads a program-uniform scalar: the program-scalar
    producer and `store_at`'s per-program base. Keeping one emitter is what
    guarantees "one load per program, never re-derived per lane".
    """
    from compiler.backend.triton_backend import DTYPE_TO_TRITON, _view_offset

    pointer = _parameter_for(pairs, "tensor", table)
    if pointer is None:
        raise CompileError(
            DiagnosticKind.INTERNAL,
            f"program uniform scalar '{hint}' reads table '{table}', but the "
            f"table is not an external operand of its kernel; a program-uniform "
            f"load needs the table pointer in the ABI",
            pipeline=graph.pipeline,
        )
    source = (index_source_override if index_source_override is not None
              else index_source(spec, pairs))
    address = _view_offset(table, source, layouts, names, lines)
    variable = names.fresh(f"{hint}_scalar")
    # One scalar load: no mask (the host proves the index in bounds for every
    # index it can see) and no lane dependence in the address. `.to` pins the
    # declared integer dtype so address arithmetic cannot silently widen.
    lines.append(
        f"    {variable} = tl.load({pointer} + {address})"
        f".to({DTYPE_TO_TRITON[spec.dtype]})"
    )
    # The host bounds check needs the launch extent of the *axis the index
    # reads*. A flat kernel passes one extent; a 2-D grid passes one per axis.
    index = spec.index
    if hasattr(index, "maximum"):
        axis = index.axis
        extent = grid_count[axis] if isinstance(grid_count, tuple) else grid_count
    else:
        extent = grid_count
    bound = dict(table=table, index=index, dtype=spec.dtype,
                 grid_count=extent,
                 # The IR's "can the host prove this index is in the table"
                 # answer decides whether `host_guards` emits a check or
                 # deliberately emits none.  Keeping the property live here is
                 # what makes the two definitions one contract (re-review
                 # 2026-10-26: it was a dead property).
                 host_checkable=spec.host_checkable)
    return variable, bound


def emit_load(node, graph, names, pairs, layouts, lines, *, grid_count):
    """Emit the program-scalar producer's one load (see `emit_uniform_load`)."""
    spec = node.attributes.program_scalar
    return emit_uniform_load(spec, node.inputs[0], node.output, graph, names,
                             pairs, layouts, lines, grid_count=grid_count)


def fold_base(base, lane):
    """`scalar_base + affine(lane)`: the only address shape this feature forms.

    There is no arithmetic hidden here: the scalar is a program-uniform value
    and the lane part is the emitter's existing affine index expression, so the
    result is affine by construction. A lowering that wants a data-derived
    *coefficient* or a per-lane division is not expressible through this helper.
    """
    return f"({base}) + ({lane})"


def host_guards(bounds, parameters):
    """Fail-closed host checks for every table index the host can see.

    A program-id index is bounded by the launch geometry (`grid_count`), a
    literal is known now and a host scalar is a call argument; the table's own
    1-D extent is read from the caller's tensor at launch. A device scalar
    index cannot be checked here -- the IR only admits it under an explicit
    ``bounds="trusted"`` contract, and this function deliberately emits no check
    for it rather than inventing one the host cannot perform.
    """
    lines = []
    for bound in bounds:
        if not bound.get("host_checkable", True):
            # `bounds="trusted"` with a device-scalar index: the caller owns the
            # bound and the host has nothing to check.  The IR property and this
            # guard must agree, so the decision is read from one place.
            continue
        table = bound["table"]
        table_variable = parameters.get(table)
        if table_variable is None:
            raise CompileError(
                DiagnosticKind.INTERNAL,
                f"program scalar reads table '{table}', which is not a declared "
                f"input of the pipeline; the host has no tensor to bound the "
                f"index against",
            )
        index = bound["index"]
        guard = None
        if hasattr(index, "maximum"):
            guard = index.maximum(bound["grid_count"])
        elif type(index) is int:
            guard = str(index)
        elif getattr(index, "host_visible", False):
            scalar = parameters.get(index.name)
            if scalar is None:
                # A host scalar input is a declared input; if it is not in the
                # parameter map the IR and the pipeline disagree.
                raise CompileError(
                    DiagnosticKind.INTERNAL,
                    f"program scalar index '{index.name}' is a host scalar but "
                    f"not a declared pipeline input",
                )
            lines += [
                f"    if not 0 <= {scalar} < {table_variable}.shape[0]:",
                f"        raise ValueError('program_scalar index {index.name} is "
                f"outside table {table}')",
            ]
            continue
        if guard is None:
            # A device scalar under bounds="trusted": the caller's contract is
            # the only bound; no host check exists. This is stated, not silent.
            continue
        lines += [
            f"    if {guard} >= {table_variable}.shape[0]:",
            f"        raise ValueError('program_scalar index for table {table} "
            f"is out of bounds: the launch can reach ' + str({guard}) + "
            f"' but the table has ' + str({table_variable}.shape[0]) + ' entries')",
        ]
    return lines
