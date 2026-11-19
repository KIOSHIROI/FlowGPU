"""Frontend-neutral program AST consumed by the IR builder.

Phase 1b moved the statement/expression nodes here.  A frontend's job is to
produce this tree from its surface syntax; the DSL parser remains in
``flowgpu.compiler.frontend`` and produces the same classes through a compatibility
re-export.  Consequently ``flowgpu.compiler.ir`` can lower a program without importing
any frontend module.
"""
from dataclasses import dataclass, field

from flowgpu.compiler.core.position import Position
from flowgpu.compiler.core.literals import DTypeLiteral, IntLiteral, Name, ShapeLiteral  # noqa: F401
from flowgpu.compiler.core.types import TensorType, format_shape, is_symbol  # noqa: F401
from flowgpu.compiler.core.scalars import ScalarType


@dataclass
class PoisonType:
    """A poisoned type returned for a value whose definition failed.

    17.3's error recovery: when a statement's definition fails, its target
    name is bound to a PoisonType instead of being left undefined. Any later
    expression consuming this name evaluates to PoisonType silently, avoiding
    cascade diagnostics while allowing genuinely independent subsequent
    errors to be detected and reported.
    """

    def __str__(self):
        return "<poison>"


def is_poisoned(type_obj) -> bool:
    return isinstance(type_obj, PoisonType)
    
@dataclass
class Call:
    name: str
    args: list[object]
    position: Position | None = field(default=None, compare=False, repr=False)

@dataclass
class InputDecl:
    name: str
    type: TensorType
    position: Position | None = field(default=None, compare=False, repr=False)

@dataclass
class Assignment:
    """One statement binding one or more names to one operator call.

    v0.1 always bound one name: every operator produced exactly one value,
    and "every value has exactly one producer" was structural. v0.2 adds
    multi-value operators (split); `names` keeps declaration order, which
    is also the operator's output order -- the only contract a caller can
    rely on, since names are the language's sole way to select a result.
    """

    names: list[str]
    value: object
    position: Position | None = field(default=None, compare=False, repr=False)

    @property
    def name(self) -> str:
        """v0.1's field, kept for the single-name case.

        It raises for a genuinely multi-name statement rather than silently
        returning the first: a caller reaching for `.name` on `a, b = ...`
        is assuming the old invariant, and that assumption should break
        loudly at the site that made it.
        """
        if len(self.names) != 1:
            raise AttributeError(
                f"multi-name assignment {', '.join(self.names)} has no "
                "single .name"
            )
        return self.names[0]

@dataclass
class OutputDecl:
    name: str
    position: Position | None = field(default=None, compare=False, repr=False)

@dataclass
class PipelineDecl:
    name: str
    statements: list[object]
    position: Position | None = field(default=None, compare=False, repr=False)
    
@dataclass
class Symbol:
    name: str
    kind: str
    type: TensorType | None = None
