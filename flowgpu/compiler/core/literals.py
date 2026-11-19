"""Frontend-neutral expression literals: shape, dtype and static integer.

The DSL AST and the operator type rules both use these; moving them to core in
Phase 1b lets the type rules live below the frontend.  The AST module re-exports
the same classes, so isinstance and existing imports are unchanged.
"""
from dataclasses import dataclass, field

from flowgpu.compiler.core.position import Position
from flowgpu.compiler.core.scalars import ScalarType
from flowgpu.compiler.core.types import format_shape


@dataclass
class ShapeLiteral:
    """A shape in expression position: reshape's second argument.

    It is not a tensor and has no dtype; anywhere except that one argument
    position, the type checker rejects it by name. Keeping it a distinct
    node -- rather than letting the grammar's `shape` rule leak tensor
    types into expressions -- means the AST never has a half-type.
    """

    shape: list[str]
    position: Position | None = field(default=None, compare=False, repr=False)

    def __str__(self):
        return format_shape(self.shape)

@dataclass
class DTypeLiteral:
    """A dtype in expression position: cast_to's second argument.

    It is not a tensor and has no shape; anywhere except that one argument
    position, the type checker rejects it by name. The target has to be
    written, not inferred from another value, so this literal is what replaces
    the `[1]` dtype-template tensor a program used to need (`cast_like`):
    `cast_to(res, bf16)` states the dtype and nothing else exists to carry it.
    """

    dtype: str
    position: Position | None = field(default=None, compare=False, repr=False)

    def __str__(self):
        return self.dtype


@dataclass
class Name:
    name: str
    # Excluded from equality: a position says where a node was written, not
    # which node it is. `x - r` and `sub(x, r)` are the same AST at different
    # columns, and the test that holds that claim compares them directly.
    position: Position | None = field(default=None, compare=False, repr=False)


@dataclass
class IntLiteral:
    """A typed static integer argument, never a runtime tensor or dimension."""
    value: int
    position: Position | None = field(default=None, compare=False, repr=False)

    @property
    def type(self) -> ScalarType:
        return ScalarType("int32")

