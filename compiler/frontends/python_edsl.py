"""A small Python eDSL that builds the core AST directly.

This is Phase 2a of the de-DSL work: a user can describe a FlowGPU pipeline in
ordinary Python and hand the result to the full compiler without serialising to
DSL text or importing the DSL lexer/parser.

    from compiler.frontends.python_edsl import program
    from compiler.compile import compile_pipelines

    with program("p") as p:
        x = p.input("x", "fp32[N]")
        y = p.relu(x)
        z = y + x
        p.output(z)

    result = compile_pipelines(p.pipelines)

The builder emits ``compiler.core.ast`` nodes (the same nodes the DSL parser
produces), so type inference, dataflow lowering, fusion, liveness and the
backends are shared rather than reimplemented.  Operator names are checked
against the core registry; type/shape errors are left to the ordinary
TypeInference stage so both frontends get identical diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from compiler.core.ast import (
    Assignment,
    Call,
    InputDecl,
    Name,
    OutputDecl,
    PipelineDecl,
)
from compiler.core.literals import DTypeLiteral, IntLiteral, ShapeLiteral
from compiler.core.operators import OPERATORS
from compiler.core.position import Position
from compiler.core.types import TensorType


def T(text: str) -> TensorType:
    """``T(\"fp32[N]\")`` / ``T(\"fp32\")`` -- a readable type constructor."""
    text = str(text).strip()
    if "[" not in text:
        return TensorType(dtype=text, shape=[])
    dtype, rest = text.split("[", 1)
    shape = [part.strip() for part in rest.rstrip("]").split(",") if part.strip()]
    return TensorType(dtype=dtype.strip(), shape=shape)


@dataclass
class Value:
    """A named value in the builder; operators return new Values."""

    name: str
    _program: "ProgramBuilder" = field(repr=False, compare=False)

    def __add__(self, other):
        return self._program.call("add", self, other)

    def __radd__(self, other):
        return self._program.call("add", other, self)

    def __sub__(self, other):
        return self._program.call("sub", self, other)

    def __rsub__(self, other):
        return self._program.call("sub", other, self)

    def __mul__(self, other):
        return self._program.call("mul", self, other)

    def __rmul__(self, other):
        return self._program.call("mul", other, self)

    def __repr__(self) -> str:
        return f"Value({self.name!r})"


class ProgramBuilder:
    """Collects core AST statements; use as a context manager."""

    def __init__(self, name: str):
        self.name = name
        self._statements: list[object] = []
        self._defined: set[str] = set()
        self._counter = 0
        self._closed = False

    # -- context ---------------------------------------------------------
    def __enter__(self) -> "ProgramBuilder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._closed = True

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this program builder is already closed")

    def _fresh(self, hint: str) -> str:
        while True:
            self._counter += 1
            name = f"{hint}{self._counter}"
            if name not in self._defined:
                return name

    # -- statements ------------------------------------------------------
    def input(self, name: str, type_text: str | TensorType) -> Value:
        self._check_open()
        if name in self._defined:
            raise ValueError(f"input {name!r} is already defined")
        declared = type_text if isinstance(type_text, TensorType) else T(type_text)
        self._defined.add(name)
        self._statements.append(InputDecl(name, declared))
        return Value(name, self)

    def output(self, value: Value | str) -> None:
        self._check_open()
        name = value.name if isinstance(value, Value) else str(value)
        if name not in self._defined:
            raise ValueError(f"output {name!r} is not defined")
        self._statements.append(OutputDecl(name))

    def call(self, op: str, *args, out: str | None = None) -> Value:
        """Emit one operator call; returns its (possibly generated) result."""
        self._check_open()
        if op not in OPERATORS:
            raise ValueError(f"unknown operator {op!r}; register it in the core operator table")
        literal_args = []
        for arg in args:
            if isinstance(arg, Value):
                literal_args.append(Name(arg.name))
            elif isinstance(arg, (IntLiteral, DTypeLiteral, ShapeLiteral)):
                literal_args.append(arg)
            elif isinstance(arg, int):
                literal_args.append(IntLiteral(arg))
            else:
                raise TypeError(f"{op}: unsupported argument {arg!r}; use a Value or p.int(...)")
        result = out or self._fresh("t")
        if result in self._defined:
            raise ValueError(f"result name {result!r} is already defined")
        self._defined.add(result)
        self._statements.append(Assignment([result], Call(op, literal_args)))
        return Value(result, self)

    # -- literal helpers -------------------------------------------------
    def int(self, value: int) -> IntLiteral:
        return IntLiteral(int(value))

    def dtype(self, name: str) -> DTypeLiteral:
        return DTypeLiteral(str(name))

    def shape(self, *dimensions: str | int) -> ShapeLiteral:
        return ShapeLiteral([str(dimension) for dimension in dimensions])

    # -- convenience operators (no need to spell p.call) -----------------
    def relu(self, value: Value) -> Value:
        return self.call("relu", value)

    def sigmoid(self, value: Value) -> Value:
        return self.call("sigmoid", value)

    def silu(self, value: Value) -> Value:
        return self.call("silu", value)

    def cast_to(self, value: Value, dtype: str) -> Value:
        return self.call("cast_to", value, self.dtype(dtype))

    def __getattr__(self, op: str):
        # Unknown attributes become operator calls only when the operator is
        # registered; ordinary attribute errors stay ordinary.
        if op.startswith("_") or op not in OPERATORS:
            raise AttributeError(op)
        return lambda *args, **kwargs: self.call(op, *args, **kwargs)

    # -- output ----------------------------------------------------------
    @property
    def pipelines(self) -> tuple[PipelineDecl, ...]:
        if not any(isinstance(statement, OutputDecl) for statement in self._statements):
            raise ValueError(f"program {self.name!r} has no output")
        return (PipelineDecl(self.name, list(self._statements)),)

    def build(self) -> tuple[PipelineDecl, ...]:
        return self.pipelines

    def compile(self, **options):
        """Convenience: compile the built pipelines (imported lazily)."""
        from compiler.compile import compile_pipelines
        return compile_pipelines(self.pipelines, **options)


def program(name: str) -> ProgramBuilder:
    return ProgramBuilder(name)
