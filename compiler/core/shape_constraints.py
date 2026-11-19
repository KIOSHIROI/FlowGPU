"""Small, explicit constraints for symbolic tensor dimensions.

The language still has no dimension-expression syntax.  This module therefore
keeps constraints outside the source grammar and lets callers validate concrete
``shape_bindings`` without changing the existing TensorType representation.
"""

from dataclasses import dataclass
from numbers import Integral


_OPERATORS = frozenset({"equal", "divisible", "ge"})


def _dimension(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer or symbol name")
    if isinstance(value, Integral):
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value
    if isinstance(value, str) and value and value.isidentifier():
        return value
    raise ValueError(f"{name} must be a positive integer or symbol name")


@dataclass(frozen=True)
class ShapeConstraint:
    """One relation between dimension literals or symbolic names.

    Use the named constructors rather than constructing this dataclass directly:
    ``equal("M", "N")``, ``divisible("D", 8)`` and ``ge("N", 1)``.
    """

    operator: str
    left: int | str
    right: int | str

    def __post_init__(self):
        if self.operator not in _OPERATORS:
            raise ValueError(f"unknown shape constraint operator: {self.operator}")
        object.__setattr__(self, "left", _dimension(self.left, "left"))
        object.__setattr__(self, "right", _dimension(self.right, "right"))
        if self.operator == "divisible" and isinstance(self.right, int) and self.right == 0:
            raise ValueError("divisibility modulus must be positive")

    @classmethod
    def equal(cls, left, right):
        return cls("equal", left, right)

    @classmethod
    def divisible(cls, left, right):
        return cls("divisible", left, right)

    @classmethod
    def ge(cls, left, right):
        return cls("ge", left, right)

    @property
    def symbols(self):
        return tuple(
            value for value in (self.left, self.right) if isinstance(value, str)
        )

    def check(self, bindings):
        """Return whether this constraint holds for the available bindings.

        ``None`` means that a symbolic operand is not bound yet; compilation
        validates the constraint's shape but deliberately defers the check.
        """
        values = []
        for operand in (self.left, self.right):
            if isinstance(operand, str):
                if operand not in bindings:
                    return None
                values.append(bindings[operand])
            else:
                values.append(operand)
        left, right = values
        if self.operator == "equal":
            return left == right
        if self.operator == "divisible":
            return left % right == 0
        return left >= right

    def describe(self):
        symbol = {"equal": "==", "divisible": "%", "ge": ">="}[self.operator]
        if self.operator == "divisible":
            return f"{self.left} % {self.right} == 0"
        return f"{self.left} {symbol} {self.right}"


class ShapeConstraints:
    """Validated collection of symbolic shape constraints."""

    def __init__(self, constraints=()):
        self.constraints = tuple(self._coerce(item) for item in constraints)

    @staticmethod
    def _coerce(item):
        if isinstance(item, ShapeConstraint):
            return item
        if isinstance(item, dict):
            try:
                return ShapeConstraint(
                    item["operator"], item["left"], item["right"]
                )
            except KeyError as error:
                raise ValueError(
                    "shape constraint dictionaries need operator, left, and right"
                ) from error
        raise ValueError(
            "shape_constraints must contain ShapeConstraint objects or dictionaries"
        )

    @property
    def symbols(self):
        return frozenset(
            symbol for constraint in self.constraints for symbol in constraint.symbols
        )

    def validate(self, *, known_symbols=(), bindings=None):
        """Validate references and any relation decidable from ``bindings``."""
        unknown = self.symbols - set(known_symbols)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown shape symbols in constraints: {names}")
        if bindings is None:
            return
        for constraint in self.constraints:
            result = constraint.check(bindings)
            if result is False:
                raise ValueError(
                    f"shape constraint violated: {constraint.describe()}"
                )


def equal(left, right):
    return ShapeConstraint.equal(left, right)


def divisible(left, right):
    return ShapeConstraint.divisible(left, right)


def ge(left, right):
    return ShapeConstraint.ge(left, right)
