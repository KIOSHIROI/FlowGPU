import pytest

from compiler.compile import CompileError
from compiler.diagnostic import DiagnosticKind
from compiler.parser import parse_ast
from compiler.symbol_table_builder import SymbolTableBuilder
from compiler.type_inference import TypeInference


def setup(source: str):
    pipeline = parse_ast(source)[0]  # a file is pipeline+
    table = SymbolTableBuilder().build(pipeline)
    return pipeline, table


def test_add_inference():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input residual : bf16[N,D]

        y = x + residual

        output y
    }
    """

    pipeline, table = setup(source)

    assignment = pipeline.statements[2]

    inferred = TypeInference(table).infer(
        assignment.value
    )

    assert inferred.dtype == "bf16"
    assert inferred.shape == ["N", "D"]
    
def test_add_shape_mismatch():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input residual : bf16[N,H]

        y = x + residual

        output y
    }
    """

    # Incompatible shapes are now rejected while the body is checked, i.e.
    # inside build(). The table is never handed out half-checked.
    with pytest.raises(CompileError, match="shapes") as caught:
        setup(source)

    assert caught.value.kind is DiagnosticKind.SHAPE

def test_rms_norm_inference():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input weight : bf16[D]

        y = rms_norm(x, weight)

        output y
    }
    """

    pipeline, table = setup(source)

    assignment = pipeline.statements[2]

    inferred = TypeInference(table).infer(
        assignment.value
    )

    assert inferred.dtype == "bf16"
    assert inferred.shape == ["N", "D"]


def test_inferred_type_is_written_back():
    """Regression: inferred types must land in the symbol table.

    Assignments used to be bound with type=None while TypeInference refused
    to read a symbol without a type, so *any* pipeline with two chained
    assignments died with "Symbol 'y' has no type". Design.md 12.1 is
    exactly such a pipeline, and this test takes its shape.
    """
    source = """
    pipeline residual_norm {
        input x        : bf16[N,D]
        input residual : bf16[N,D]
        input weight   : bf16[D]

        y = add(x, residual)
        z = rms_norm(y, weight)

        output z
    }
    """

    _ast, table = setup(source)

    assert table.lookup("y").type is not None
    assert table.lookup("y").type.dtype == "bf16"
    assert table.lookup("y").type.shape == ["N", "D"]

    assert table.lookup("z").type is not None
    assert table.lookup("z").type.shape == ["N", "D"]


def test_forward_reference_is_rejected():
    """The body is dataflow: a value exists only after its producer."""
    source = """
    pipeline test {
        y = rms_norm(x, w)
        input x : bf16[N,D]
        input w : bf16[D]
        output y
    }
    """

    with pytest.raises(
        CompileError, match="Undefined symbol 'x'"
    ) as caught:
        setup(source)

    assert caught.value.kind is DiagnosticKind.DEPENDENCY


def test_sub_inference():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input r : bf16[N,D]

        y = x - r

        output y
    }
    """

    pipeline, table = setup(source)

    inferred = TypeInference(table).infer(pipeline.statements[2].value)

    assert inferred.dtype == "bf16"
    assert inferred.shape == ["N", "D"]


def test_sub_shape_mismatch():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input r : bf16[N,H]

        y = x - r

        output y
    }
    """

    with pytest.raises(CompileError, match="shapes") as caught:
        setup(source)

    assert caught.value.kind is DiagnosticKind.SHAPE


def test_infix_and_function_forms_infer_the_same_type():
    """`x - r` 和 `sub(x, r)` 必须走同一条类型规则。

    它们此前是两套实现：一个 Add 节点带一份自己的检查逻辑，注册表里
    另有一份逐字相同的 infer_add。中缀降级为算子调用之后，同一语义只剩
    一处实现，这个断言保护的就是这一点。
    """
    def inferred_type_of(expression):
        pipeline, table = setup(f"""
        pipeline test {{
            input x : bf16[N,D]
            input r : bf16[N,D]

            y = {expression}

            output y
        }}
        """)

        return TypeInference(table).infer(pipeline.statements[2].value)

    assert inferred_type_of("x - r") == inferred_type_of("sub(x, r)")
    assert inferred_type_of("x * r") == inferred_type_of("mul(x, r)")
    assert inferred_type_of("x + r") == inferred_type_of("add(x, r)")


def test_relu_inference():
    """Design.md 3 (P3) calls relu; it preserves dtype and shape."""
    source = """
    pipeline test {
        input x : bf16[B,H,S,D]

        y = relu(x)

        output y
    }
    """

    pipeline, table = setup(source)

    inferred = TypeInference(table).infer(pipeline.statements[1].value)

    assert inferred.dtype == "bf16"
    assert inferred.shape == ["B", "H", "S", "D"]


def test_unary_operators_preserve_dtype_and_shape():
    """One rule for every unary operator whose result has its operand's type.

    silu, relu and softmax preserve the input type; floating operators add dtype checks.
    A type rule describes the operand type and the result type, and for these
    three those are equal -- which is why adding an operator that is a new
    instance of an existing rule shape costs no rule code at all.
    """
    def inferred_type_of(operator):
        source = f"""
        pipeline test {{
            input x : bf16[B,H,S,D]

            y = {operator}(x)

            output y
        }}
        """

        pipeline, table = setup(source)

        return TypeInference(table).infer(pipeline.statements[1].value)

    types = {
        name: inferred_type_of(name)
        for name in ("silu", "relu", "softmax")
    }

    assert set(map(str, types.values())) == {"bf16[B,H,S,D]"}, types


def test_relu_arity_is_checked():
    source = """
    pipeline test {
        input x : bf16[N,D]

        y = relu(x, x)

        output y
    }
    """

    with pytest.raises(CompileError, match="expects 1 argument") as caught:
        setup(source)

    assert caught.value.kind is DiagnosticKind.OPERATOR
