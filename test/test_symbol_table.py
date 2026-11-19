import pytest

from flowgpu.compiler.compile import CompileError
from flowgpu.compiler.diagnostic import DiagnosticKind
from flowgpu.compiler.parser import parse_ast
from flowgpu.compiler.symbol_table_builder import SymbolTableBuilder


def build(source: str):
    pipeline = parse_ast(source)[0]  # a file is pipeline+
    return SymbolTableBuilder().build(pipeline)


def test_build_symbol_table():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input weight : bf16[D]

        y = x
        output y
    }
    """

    table = build(source)

    assert table.contains("x")
    assert table.contains("weight")
    assert table.contains("y")


def test_input_symbol():
    source = """
    pipeline test {
        input x : bf16[N,D]
    }
    """

    table = build(source)

    symbol = table.lookup("x")

    assert symbol.name == "x"
    assert symbol.kind == "input"
    assert symbol.type.dtype == "bf16"
    assert symbol.type.shape == ["N", "D"]


def test_undefined_symbol():
    source = """
    pipeline test {
        input x : bf16[N,D]

        y = x + unknown

        output y
    }
    """

    with pytest.raises(
        CompileError, match="Undefined symbol 'unknown'"
    ) as caught:
        build(source)

    assert caught.value.kind is DiagnosticKind.DEPENDENCY


def test_undefined_output():
    source = """
    pipeline test {
        input x : bf16[N,D]

        output y
    }
    """

    with pytest.raises(
        CompileError, match="Undefined symbol 'y'"
    ) as caught:
        build(source)

    assert caught.value.kind is DiagnosticKind.DEPENDENCY
    # The position is the output statement, not the pipeline.
    assert caught.value.line == 5
        
def test_duplicate_definition():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input x : bf16[N,D]
    }
    """

    with pytest.raises(
        CompileError,
        match="Symbol 'x' is already defined",
    ) as caught:
        build(source)

    assert caught.value.kind is DiagnosticKind.DEPENDENCY