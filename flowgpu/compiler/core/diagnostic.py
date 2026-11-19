"""Structured diagnostics -- what Design.md 17.3 promises.

Moved to the frontend-neutral ``flowgpu.compiler.core`` package in Phase 0 of the
de-DSL work; ``flowgpu.compiler.frontend.diagnostic`` re-exports these classes.

The compiler used to decide the category of a semantic error from the Python
exception class it caught: ValueError meant a name error and TypeError meant
a type error, so a dtype mismatch, a shape mismatch, a wrong argument count
and a reserved operator all arrived as the same "type" phase. The category
was lost at the raise site, which is the only place that knows it.

Two kinds of object replace that, and the split is the point:

* `RuleError` is raised by the layers that only know *what* went wrong: an
  operator's type rule in operator.py, the symbol table. They are pure
  functions over types and names and stay that way -- no source positions, so
  they remain trivially unit-testable.

* `CompileError` is raised by the layers that know *where*: the passes that
  walk the AST hold the node, so they attach its position to a RuleError and
  turn it into a CompileError. This is the one error type a caller sees.
"""

from enum import Enum

from flowgpu.compiler.core.position import Position


class DiagnosticKind(str, Enum):
    """The category of a diagnostic.

    SYNTAX is not in Design.md 17.3, which speaks about semantic feedback;
    it is the fifth category because a file that does not parse still has to
    be reported, and pretending it is one of the four would be a lie.

    INTERNAL classifies malformed compiler IR separately from source errors.
    The four semantic categories are that section's list; their values are the
    lower-case form used in messages and in structured output:

        DEPENDENCY -> DependencyError
        TYPE       -> TypeError
        SHAPE      -> ShapeError
        OPERATOR   -> OperatorError
    """

    SYNTAX = "syntax"
    DEPENDENCY = "dependency"
    TYPE = "type"
    SHAPE = "shape"
    OPERATOR = "operator"
    # Malformed compiler IR, rather than a source error for users to repair.
    INTERNAL = "internal"


class RuleError(Exception):
    """A type rule, or the symbol table, refused something.

    Position-free on purpose: a rule knows the operand types and not the
    file, so the position is added by the pass that called it.
    """

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = DiagnosticKind(kind)
        self.message = message


class CompileError(Exception):
    """A compilation failure, ready to be shown or serialised.

    kind     source failure category, or INTERNAL for a compiler IR defect
    message  what is wrong, in the language's own terms
    position where it is, when the stage that failed knew (Position | None)
    pipeline which pipeline of the file, when that is known
    others   what the same pass found after this one, before it gave up

    `others` is the whole of 17.3's improvement. A pass that raises on the first
    bad statement hands the repair loop one thing to fix, so a file with four
    independent mistakes costs four compiles. The pass now walks every statement
    and reports what it found, and *this* error is still the one that stopped it:
    the first in source order, with the rest travelling alongside.
    """

    def __init__(self, kind, message, position=None, pipeline=None, others=()):
        super().__init__(message)
        self.kind = DiagnosticKind(kind)
        self.message = message
        self.position = position
        self.pipeline = pipeline
        self.others = tuple(others)

    def all_diagnostics(self):
        """This diagnostic and everything found after it, flattened.

        Flattened because collectors nest: a pipeline collects what its
        statements found, and the file collects what its pipelines found.
        """
        found = [self]

        for other in self.others:
            found.extend(other.all_diagnostics())

        return tuple(found)

    def as_dicts(self):
        return [diagnostic.as_dict() for diagnostic in self.all_diagnostics()]

    @property
    def line(self):
        return self.position.line if self.position is not None else None

    @property
    def column(self):
        return self.position.column if self.position is not None else None

    def with_pipeline(self, name):
        """The same diagnostic, now known to come from pipeline `name`."""
        return CompileError(
            kind=self.kind,
            message=self.message,
            position=self.position,
            pipeline=name,
            others=self.others,
        )

    def as_dict(self):
        """The diagnostic as JSON-compatible data.

        This is the shape Design.md 17.4's repair loop consumes: the kind to
        decide what to fix, the position to decide where, and the message to
        hand to the model.
        """
        return {
            "pipeline": self.pipeline,
            "kind": self.kind.value,
            "message": self.message,
            "line": self.line,
            "column": self.column,
        }

    def __str__(self):
        where = ""

        if self.pipeline is not None:
            where = f" in pipeline '{self.pipeline}'"

        if self.position is not None:
            where += f" at {self.position}"

        return f"{self.kind.value} error{where}: {self.message}"
