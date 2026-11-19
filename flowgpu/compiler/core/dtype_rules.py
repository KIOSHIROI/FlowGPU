"""21.6 dtype and precision rules, owned by the frontend-neutral core.

The frontend enforces these at the call site (type_inference and the
operator rules), and the analysis layer reads the same tables for its
budgets.  They live below `analysis` because `frontend` may not import a
layer above it (review P2, layering, 2026-10-26).
"""

from flowgpu.compiler.core.diagnostic import DiagnosticKind, RuleError

# The scalar types of 8.1 and their widths. This is the one piece of backend
# knowledge that checking needs, because rule 2 compares precisions by width.
# It lived in liveness.py while 11's `size` was its only reader; now that a rule
# reads it, a rule module owns it and liveness imports it from here.
DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp8e4m3": 1,
    "fp8e5m2": 1,
    "int8": 1,
    "uint8": 1,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}

# 21.6's classes. The class decides whether an operator may be applied at all,
# which the dtype alone cannot: `bool + bool` has matching operands and no
# arithmetic.
#: Narrow storage dtypes.  They exist so a kernel can *store* quantised results --
#: CANN-Bench's DynamicQuant and the quantised matmuls produce int8, the FP8
#: operators produce e4m3 -- not so the language grows 8-bit arithmetic.  A
#: program must widen to a real compute dtype to do anything with the values, and
#: `require_arithmetic` enforces that: the class is "storage", which is neither a
#: float nor an integer.
NARROW_INT_DTYPES = frozenset({"int8", "uint8"})
STORAGE_DTYPES = frozenset({"int8", "uint8", "fp8e4m3", "fp8e5m2"})

DTYPE_CLASSES = {
    "fp16": "float",
    "bf16": "float",
    "fp32": "float",
    # Design 21.6 / value layer: int8 and uint8 are storage, not arithmetic
    # (review P2, 2026-10-26: this table said "integer" while the comment below
    # and Design.md said storage-only, so `int8 + int8` compiled).
    **{dtype: "storage" for dtype in NARROW_INT_DTYPES},
    "fp8e4m3": "storage",
    "fp8e5m2": "storage",
    "int32": "integer",
    "int64": "integer",
    "bool": "bool",
}

# Rule 2's floor. A float reduction below this is refused, whatever the
# operator asked for.
ACC_FLOOR = "fp32"


def dtype_class(dtype: str) -> str:
    """Which of 21.6's three classes a dtype belongs to."""
    if dtype not in DTYPE_CLASSES:
        raise KeyError(f"unknown dtype {dtype!r}")

    return DTYPE_CLASSES[dtype]


def width(dtype: str) -> int:
    """The dtype's width in bytes, which is how rule 2 compares precisions."""
    if dtype not in DTYPE_BYTES:
        raise KeyError(f"unknown dtype {dtype!r}")

    return DTYPE_BYTES[dtype]


def is_float(dtype: str) -> bool:
    return dtype_class(dtype) == "float"


def require_arithmetic(tensor_type, verb: str):
    """A number is needed here, and `bool` is not one.

    The dtype rule alone accepts `bool + bool`: the operands match. This is the
    class rule that 21.6 calls the degenerate case of the precision dimension,
    and it is what closes README's known gap 2.
    """
    dtype = tensor_type.dtype
    kind = dtype_class(dtype)
    if kind == "bool":
        raise RuleError(
            DiagnosticKind.TYPE,
            f"Cannot {verb} a {dtype}: bool is not a number",
        )
    if kind == "storage":
        raise RuleError(
            DiagnosticKind.TYPE,
            f"Cannot {verb} a {dtype}: it is a storage dtype, so it has to be "
            f"widened to a compute dtype first; the repair is "
            f"cast_to(x, fp32) (or bp16/bf16) before the arithmetic and "
            f"cast_to(result, {dtype}) after it",
        )


def require_numeric(tensor_type, verb: str):
    """A number is needed, but a stored one is allowed.

    Casting is how a storage dtype gets in and out of a program, so the cast
    rules must accept it even though arithmetic may not.
    """
    if dtype_class(tensor_type.dtype) == "bool":
        raise RuleError(DiagnosticKind.TYPE,
                        f"Cannot {verb} a {tensor_type.dtype}: bool is not a number")

    return tensor_type


def check_accumulation(name: str, spec, arguments) -> None:
    """Rule 2, at the call site where the operand dtypes are known."""
    if spec.acc is None:
        return

    if spec.acc not in DTYPE_BYTES:
        raise AssertionError(
            f"Operator '{name}' declares acc={spec.acc!r}, which is not a "
            f"dtype this compiler knows; 21.6 compares precisions by width, so "
            f"a declaration it cannot measure is a declaration it cannot honour"
        )

    if not any(hasattr(argument, 'dtype') and is_float(argument.dtype) for argument in arguments):
        return

    if width(spec.acc) >= width(ACC_FLOOR):
        return

    raise RuleError(
        DiagnosticKind.TYPE,
        f"Operator '{name}' accumulates float inputs in {spec.acc}, which is "
        f"narrower than {ACC_FLOOR}: 21.6 requires the accumulator of a "
        f"reduction over floats to be at least {ACC_FLOOR}",
    )


def declared(operators) -> tuple:
    """The operators that declared an accumulator, in registry order."""
    return tuple(
        name for name, spec in operators.items() if spec.acc is not None
    )


def declined(operators) -> tuple:
    """21.6's metric, both halves of it.

    The metric is the count of undeclared reductions, and there is a limit worth
    stating plainly: "this operator is a reduction" is not statically inferable
    from the registry. `transpose` is an anchor too -- `pointwise=False` -- and
    it accumulates nothing, so `acc=None` is the right answer for it rather than
    a forgotten declaration. What the contract can do is make every operator
    state a position and make the two groups short enough to inspect:

        declared(operators)   accumulates in this dtype
        declined(operators)   states that it accumulates in nothing

    Rule 1 (the required keyword) is what stops silence from appearing in the
    first group, and test_precision.py pins both groups by name -- so a new
    anchor arrives as a changed test rather than as an absent line.
    """
    return tuple(
        name
        for name, spec in operators.items()
        if spec.acc is None and spec.pointwise is False
    )
