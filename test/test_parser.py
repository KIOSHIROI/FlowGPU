import pytest
from lark.exceptions import UnexpectedInput

from flowgpu.compiler.ast_node import Assignment
from flowgpu.compiler.parser import parse_ast, parser


def parse(source: str):
    return parser.parse(source)


# ---------------------------------------------------------
# 1. 最小合法 Flow
# ---------------------------------------------------------

def test_empty_pipeline():
    source = """
    pipeline test {
    }
    """

    parse(source)


# ---------------------------------------------------------
# 2. Input declaration
# ---------------------------------------------------------

def test_input_declaration():
    source = """
    pipeline test {
        input x : bf16[N,D]
    }
    """

    parse(source)


def test_multiple_inputs():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input residual : bf16[N,D]
        input weight : bf16[D]
    }
    """

    parse(source)


# ---------------------------------------------------------
# 3. Tensor types
# ---------------------------------------------------------

def test_tensor_types():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input a : fp16[B,H,S,D]
        input b : fp32[N]
        input c : int32[N,D]
    }
    """

    parse(source)


# ---------------------------------------------------------
# 4. Assignment
# ---------------------------------------------------------

def test_simple_assignment():
    source = """
    pipeline test {
        input x : bf16[N,D]

        y = x
    }
    """

    parse(source)


# ---------------------------------------------------------
# 5. Addition expression
# ---------------------------------------------------------

def test_addition():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input y : bf16[N,D]

        z = x + y
    }
    """

    parse(source)


def test_chained_addition():
    source = """
    pipeline test {
        input a : bf16[N,D]
        input b : bf16[N,D]
        input c : bf16[N,D]

        y = a + b + c
    }
    """

    parse(source)


# ---------------------------------------------------------
# 6. Function call
# ---------------------------------------------------------

def test_function_call():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input weight : bf16[D]

        y = rms_norm(x, weight)
    }
    """

    parse(source)


def test_function_call_multiple_arguments():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input w : bf16[D,M]

        y = matmul(x, w)
    }
    """


    parse(source)


def test_nested_function_call():
    source = """
    pipeline test {
        input x : bf16[N,D]
        input weight : bf16[D]

        y = silu(rms_norm(x, weight))
    }
    """

    parse(source)


# ---------------------------------------------------------
# 7. Output declaration
# ---------------------------------------------------------

def test_output():
    source = """
    pipeline test {
        input x : bf16[N,D]

        y = x

        output y
    }
    """

    parse(source)


# ---------------------------------------------------------
# 8. Complete FlowGPU program
# ---------------------------------------------------------

def test_complete_flowgpu_program():
    source = """
    pipeline residual_norm {

        input x : bf16[N,D]
        input residual : bf16[N,D]
        input weight : bf16[D]

        y = x + residual
        z = rms_norm(y, weight)

        output z
    }
    """

    parse(source)


def test_mlp_program():
    source = """
    pipeline mlp {

        input x : bf16[B,D]
        input w1 : bf16[D,H]
        input w2 : bf16[H,D]

        h = matmul(x, w1)
        h = silu(h)
        y = matmul(h, w2)

        output y
    }
    """

    parse(source)


# ---------------------------------------------------------
# 9. Invalid syntax
# ---------------------------------------------------------

def test_missing_colon():
    source = """
    pipeline test {
        input x bf16[N,D]
    }
    """

    with pytest.raises(UnexpectedInput, match="oken\\('DTYPE', 'bf16'\\) at lin"):
        parse(source)


def test_scalar_declaration_is_rank_zero():
    """`input x : bf16` is now a scalar declaration, not a missing shape.

    The language had no rank-0 type, so this used to be a syntax error. A
    scalar is written as a bare dtype; a bracketed shape is still required to
    have at least one dimension (`bf16[1]` stays a one-element tensor).
    """
    source = """
    pipeline test {
        input x : bf16
    }
    """

    pipeline = parse_ast(source)[0]
    (declared,) = pipeline.statements
    assert declared.type.dtype == "bf16"
    assert declared.type.shape == []
    assert declared.type.is_scalar


def test_missing_scalar_type():
    source = """
    pipeline test {
        input x : [N,D]
    }
    """

    with pytest.raises(UnexpectedInput, match="oken\\('LBRACK', '\\['\\) at lin"):
        parse(source)


def test_missing_right_bracket():
    source = """
    pipeline test {
        input x : bf16[N,D
    }
    """

    with pytest.raises(UnexpectedInput, match='Unexpected token Token'):
        parse(source)


def test_missing_equal():
    source = """
    pipeline test {
        input x : bf16[N,D]

        y x + x
    }
    """

    with pytest.raises(UnexpectedInput, match="oken\\('NAME', 'x'\\) at lin"):
        parse(source)


def test_missing_output_name():
    source = """
    pipeline test {
        input x : bf16[N,D]

        output
    }
    """

    with pytest.raises(UnexpectedInput, match='Unexpected token Token'):
        parse(source)


def test_missing_right_brace():
    source = """
    pipeline test {
        input x : bf16[N,D]
        output x
    """

    with pytest.raises(UnexpectedInput, match='Unexpected token Token'):
        parse(source)


# ---------------------------------------------------------
# 10. Invalid pipeline declaration
# ---------------------------------------------------------

def test_missing_pipeline_keyword():
    source = """
    test {
        input x : bf16[N,D]
    }
    """

    with pytest.raises(UnexpectedInput, match="oken\\('NAME', 'test'\\) at lin"):
        parse(source)


def test_missing_pipeline_name():
    source = """
    pipeline {
        input x : bf16[N,D]
    }
    """

    with pytest.raises(UnexpectedInput, match='Unexpected token Token'):
        parse(source)


# ---------------------------------------------------------
# 11. Invalid expression
# ---------------------------------------------------------

def test_invalid_expression():
    source = """
    pipeline test {
        input x : bf16[N,D]

        y = +
    }
    """

    with pytest.raises(UnexpectedInput, match="oken\\('PLUS', '\\+'\\) at lin"):
        parse(source)


def test_missing_function_argument():
    source = """
    pipeline test {
        input x : bf16[N,D]

        y = rms_norm()
    }
    """

    # 注意：
    # 当前 Grammar 允许空 arguments，
    # 所以这个测试暂时不应该作为语法错误。
    parse(source)


# ---------------------------------------------------------
# 6. 中缀算术是算子调用的语法糖
# ---------------------------------------------------------

def expression_of(expression_source: str):
    """`y = <expr>` 的 AST，包在一个最小 pipeline 里。"""
    pipeline = parse_ast(
        f"""
        pipeline test {{
            input x : bf16[N,D]
            input r : bf16[N,D]
            input s : bf16[N,D]

            y = {expression_source}

            output y
        }}
        """
    )[0]

    return [
        statement
        for statement in pipeline.statements
        if isinstance(statement, Assignment)
    ][0].value


@pytest.mark.parametrize(
    "infix,call",
    [
        ("x + r", "add(x, r)"),
        ("x - r", "sub(x, r)"),
        ("x * r", "mul(x, r)"),
    ],
)
def test_infix_is_sugar_for_the_operator_call(infix, call):
    """两种写法必须得到同一个 AST。

    Design.md 7.1 只给了一次计算一种形状 —— 以算子命名的节点。若两种写法
    产生两种节点，就等于把一个设计特意抹掉的区分塞回了语言核心，而 IR、
    类型检查、依赖分析各自都要为它多写一条分支。
    """
    assert expression_of(infix) == expression_of(call)


def test_multiplication_binds_tighter_than_addition():
    """优先级来自语法分层，不来自任何优先级声明。"""
    from flowgpu.compiler.ast_node import Call, Name

    assert expression_of("x + r * s") == Call(
        name="add",
        args=[Name("x"), Call(name="mul", args=[Name("r"), Name("s")])],
    )


def test_multiplication_on_the_left_of_a_sum_is_allowed():
    """回归：`addition` 的基例必须是 `multiplication`。

    若基例写成 `atom`，`x * r` 除了作为加号的右操作数之外没有任何合法
    位置，于是 `x * r + s` 会被直接拒绝。
    """
    parse("pipeline p { input x : bf16[N,D] input r : bf16[N,D] y = x * r output y }")


def test_subtraction_is_left_associative():
    from flowgpu.compiler.ast_node import Call, Name

    assert expression_of("x - r - s") == Call(
        name="sub",
        args=[Call(name="sub", args=[Name("x"), Name("r")]), Name("s")],
    )