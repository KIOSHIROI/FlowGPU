"""The `store_at` / `store_at_trusted` type rule, free of registry imports.

Kept apart from the staged registrar (`compiler/frontend/store_at.py`) so the
coordinated registry commit can be one import line plus two registrations in
`compiler/frontend/operator.py`, with no circular import:

    operator.py -> store_at_rule            (this module; leaf)
    store_at.py -> operator.py + store_at_rule

The contract is in ``docs/STORE_AT_DESIGN_20261022.md``: `store_at` writes the
contiguous run `values[p, :]` at `dst_flat[offsets[p] + arange(lane_count)]`,
where `offsets[p]` is a program-uniform scalar read once per program. The
`_trusted` spelling is the author-level declaration that the bases are disjoint
and in bounds (it skips the per-call host D2H check and records the declaration
in the emitted host).
"""

from compiler.frontend.ast_node import TensorType
from compiler.frontend.diagnostic import DiagnosticKind
from compiler.frontend.dtype_rules import RuleError


STORE_AT_OPERATOR = "store_at"
STORE_AT_TRUSTED_OPERATOR = "store_at_trusted"
STORE_AT_OPERATORS = (STORE_AT_OPERATOR, STORE_AT_TRUSTED_OPERATOR)


def reject(kind, message):
    """The rule-level refusal; the type checker attaches the call-site position."""
    return RuleError(kind, message)


def infer_store_at(args):
    """`dst` and `values` same dtype; `offsets` a rank-1 integer table."""
    dst, offsets, values = args
    if len(dst.shape) < 1:
        raise reject(DiagnosticKind.SHAPE,
                     "store_at needs a rank >= 1 destination")
    if len(values.shape) < 1:
        raise reject(DiagnosticKind.SHAPE, "store_at needs rank >= 1 values")
    if len(offsets.shape) != 1 or offsets.dtype not in {"int32", "int64"}:
        raise reject(
            DiagnosticKind.TYPE,
            "store_at requires a rank-1 int32/int64 offsets table (one "
            "program-uniform base per program)",
        )
    if str(values.shape[0]) != str(offsets.shape[0]):
        raise reject(
            DiagnosticKind.SHAPE,
            f"store_at needs one base per value row: values have "
            f"{values.shape[0]} rows and offsets have {offsets.shape[0]}",
        )
    if dst.dtype != values.dtype:
        raise reject(
            DiagnosticKind.TYPE,
            f"store_at requires matching dtypes: destination {dst.dtype} and "
            f"values {values.dtype}",
        )
    return TensorType(dtype=dst.dtype, shape=dst.shape.copy())
