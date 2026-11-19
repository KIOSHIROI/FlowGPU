"""Frontend-neutral type representation: tensor/scalar types and shapes.

``TensorType`` used to live in ``compiler.frontend.ast_node`` even though the
IR, analyses and backends all consume it.  Phase 0 moves it here and leaves a
compatibility re-export in the DSL AST module; a future frontend can construct
the same type without importing the DSL parser.
"""
from dataclasses import dataclass


def is_symbol(dimension) -> bool:
    """Classify a source dimension independently of any target backend."""
    return not str(dimension).isdigit()


def format_shape(shape: list[str]) -> str:
    """A shape as the language writes it: [N,D], not Python's ['N', 'D'].

    Diagnostic messages are read by a repair loop, so they stay inside the
    language: whatever appears in a message is something the user could have
    written.
    """
    return f"[{','.join(shape)}]"


@dataclass
class TensorType:
    dtype: str
    shape: list[str]

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def is_scalar(self) -> bool:
        """A declared scalar: a dtype with no shape (`input alpha:fp32`).

        Rank 0, and a different kind of value from a tensor with one element:
        `fp16[1]` still has an extent to index, while a scalar is one number
        the caller passes to the host entry. Every pass that needs a real
        tensor asks this, so the distinction is made once here rather than
        inferred from `len(shape) == 0` in each of them.
        """
        return not self.shape

    def __str__(self):
        # The language writes a scalar as a bare dtype, with no brackets.
        if self.is_scalar:
            return self.dtype

        return f"{self.dtype}{format_shape(self.shape)}"
