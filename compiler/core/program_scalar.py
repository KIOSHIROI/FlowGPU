"""Staged type rule for the program-uniform scalar producer (not registered).

This is the compiler-side half of the program-uniform scalar capability: the
IR contract lives in ``ir/program_scalar_ir``, the emission in
``backend/program_scalar``, the address contract in
``analysis/write_address``. The user-visible operator is *not* registered in
``frontend/operator.py`` yet -- that spelling and its consumer are the operator
step this compiler work unblocks, and leaving the registry untouched keeps every
default build byte-identical.

What is decided here, so the rule and the lowering cannot disagree:

* the table operand is a **rank-1 integer** tensor (`int32`/`int64`) -- a
  floating index is a cast the language does not make implicitly;
* the result is a **rank-0 integer** scalar, program-uniform: one value for
  every lane of the program, so it can be folded into a scalar base;
* the bounds contract is the IR's (``bounds="reject"``: the host proves the
  index is inside the table before launching; a device-scalar index needs an
  explicit ``trusted`` contract).

``register()`` exists for the tests of this compiler-side work and for the
operator step to call once the user-visible form is decided. It is deliberately
not called on import.
"""

from compiler.core.types import TensorType
from compiler.core.diagnostic import DiagnosticKind
from compiler.core.operators import OPERATORS, OperatorSpec, register_operator, reject

#: The staged internal name. `ir/program_scalar_ir.STAGED_PROGRAM_SCALAR_OP`
#: is asserted equal to it by the tests, so the IR, the rule and the emission
#: cannot drift about which operator carries the value category.
STAGED_OPERATOR = "program_scalar"

#: Operators whose rank-0 result is a *program-uniform value* rather than a
#: scalar that would need a buffer. `TypeInference` uses this to tell the two
#: apart: every other operator's rank-0 result is refused.
PROGRAM_UNIFORM_OPERATORS = frozenset({STAGED_OPERATOR})


def infer_program_scalar(args):
    """One rank-1 integer table in, one rank-0 integer value out."""
    (table,) = args
    if len(table.shape) != 1:
        raise reject(
            DiagnosticKind.SHAPE,
            "program_scalar reads a rank-1 table; the index is a scalar and a "
            "table with more axes would be an indexed load, not this contract",
        )
    if table.dtype not in {"int32", "int64"}:
        raise reject(
            DiagnosticKind.TYPE,
            f"program_scalar table must be int32 or int64, got {table.dtype}; "
            f"a floating index is a cast the language does not make implicitly",
        )
    return TensorType(dtype=table.dtype, shape=[])


def registration():
    """The OperatorSpec the operator step registers (tests use it directly)."""
    return OperatorSpec(
        name=STAGED_OPERATOR,
        arity=1,
        infer=infer_program_scalar,
        # It is a producer folded into the consumer kernel, so it fuses like a
        # pointwise operator: one scalar load, then one kernel for the region.
        pointwise=True,
        acc=None,
    )


def register():
    """Add the staged operator to the frontend registry (tests / operator step)."""
    if STAGED_OPERATOR not in OPERATORS:
        register_operator(
            STAGED_OPERATOR, 1, infer_program_scalar, pointwise=True, acc=None
        )
    return OPERATORS[STAGED_OPERATOR]


def unregister():
    OPERATORS.pop(STAGED_OPERATOR, None)
