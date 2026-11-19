"""Where something was written.

A position belongs to the source text, not to the AST and not to a
diagnostic: the AST carries one so that a later stage can say where it
failed, and a diagnostic carries one so that the message can point at it.
This module is frontend-neutral (Phase 0 of the de-DSL work).
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    line: int
    column: int

    @classmethod
    def from_token(cls, token) -> "Position":
        """The position of a Lark token.

        Every token carries its own line and column, so positions need no
        parser option and no propagation through the grammar: the position of
        a name is the position of the name the user wrote, which is also the
        position a diagnostic wants to point at.
        """
        return cls(line=token.line, column=token.column)

    def __str__(self) -> str:
        return f"line {self.line}, column {self.column}"
