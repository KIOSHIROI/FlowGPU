from pathlib import Path

import pytest

from flowgpu.compiler.lexer import tokenize
from lark.exceptions import UnexpectedCharacters


def tokens_of(source: str):
    return [(token.type, token.value) for token in tokenize(source)]


def test_keywords():
    source = "pipeline input output"

    assert tokens_of(source) == [
        ("PIPELINE", "pipeline"),
        ("INPUT", "input"),
        ("OUTPUT", "output"),
    ]


def test_return_is_not_a_keyword():
    """`return` was declared in both grammars but had no production."""
    assert tokens_of("return") == [("NAME", "return")]


def test_dtypes():
    source = "bf16 fp16 fp32 int32 int64 bool"

    assert tokens_of(source) == [
        ("DTYPE", "bf16"),
        ("DTYPE", "fp16"),
        ("DTYPE", "fp32"),
        ("DTYPE", "int32"),
        ("DTYPE", "int64"),
        ("DTYPE", "bool"),
    ]


def test_there_is_exactly_one_grammar():
    """The lexer must be built from the parser's grammar, not a copy.

    The keyword and dtype sets used to be duplicated in a second grammar
    file plus two Python dictionaries, and had already drifted (int64 and
    bool were missing). Sharing one file is what makes that impossible.
    """
    from flowgpu.compiler.lexer import GRAMMAR_FILE

    grammar_dir = Path(__file__).resolve().parents[1] / "flowgpu" / "compiler"

    assert GRAMMAR_FILE == grammar_dir / "grammar.lark"
    assert [path.name for path in sorted(grammar_dir.glob("*.lark"))] == [
        "grammar.lark"
    ]


def test_every_dtype_the_lexer_knows_the_parser_accepts():
    from flowgpu.compiler.parser import parse

    dtypes = [value for kind, value in tokens_of(
        "bf16 fp16 fp32 int32 int64 bool"
    ) if kind == "DTYPE"]

    assert dtypes == ["bf16", "fp16", "fp32", "int32", "int64", "bool"]

    for dtype in dtypes:
        parse(f"pipeline p {{ input x : {dtype}[N] output x }}")


def test_identifiers():
    source = "x residual rms_norm flow_gpu x123 _hidden"

    assert tokens_of(source) == [
        ("NAME", "x"),
        ("NAME", "residual"),
        ("NAME", "rms_norm"),
        ("NAME", "flow_gpu"),
        ("NAME", "x123"),
        ("NAME", "_hidden"),
    ]


def test_integers():
    source = "0 1 123 4096"

    assert tokens_of(source) == [
        ("INT", "0"),
        ("INT", "1"),
        ("INT", "123"),
        ("INT", "4096"),
    ]


def test_symbols():
    source = "{}[]():,=+-*"

    assert tokens_of(source) == [
        ("LBRACE", "{"),
        ("RBRACE", "}"),
        ("LBRACK", "["),
        ("RBRACK", "]"),
        ("LPAR", "("),
        ("RPAR", ")"),
        ("COLON", ":"),
        ("COMMA", ","),
        ("EQUAL", "="),
        ("PLUS", "+"),
        ("MINUS", "-"),
        ("STAR", "*"),
    ]


def test_keyword_identifier_boundary():
    """
    关键字只能在完整匹配时识别为关键字。
    例如 flow_gpu、input_x、bf16x 都应该是 NAME。
    """

    source = "pipeline flow_gpu input input_x bf16 bf16x"

    assert tokens_of(source) == [
        ("PIPELINE", "pipeline"),
        ("NAME", "flow_gpu"),
        ("INPUT", "input"),
        ("NAME", "input_x"),
        ("DTYPE", "bf16"),
        ("NAME", "bf16x"),
    ]


def test_complete_flowgpu_program():
    """
    使用一个真实的 FlowGPU 程序验证 Lexer。
    """

    source = """
    pipeline residual_norm {
        input x : bf16[N,D]
        input residual : bf16[N,D]

        y = x + residual

        output y
    }
    """

    assert tokens_of(source) == [
        ("PIPELINE", "pipeline"),
        ("NAME", "residual_norm"),
        ("LBRACE", "{"),

        ("INPUT", "input"),
        ("NAME", "x"),
        ("COLON", ":"),
        ("DTYPE", "bf16"),
        ("LBRACK", "["),
        ("NAME", "N"),
        ("COMMA", ","),
        ("NAME", "D"),
        ("RBRACK", "]"),

        ("INPUT", "input"),
        ("NAME", "residual"),
        ("COLON", ":"),
        ("DTYPE", "bf16"),
        ("LBRACK", "["),
        ("NAME", "N"),
        ("COMMA", ","),
        ("NAME", "D"),
        ("RBRACK", "]"),

        ("NAME", "y"),
        ("EQUAL", "="),
        ("NAME", "x"),
        ("PLUS", "+"),
        ("NAME", "residual"),

        ("OUTPUT", "output"),
        ("NAME", "y"),

        ("RBRACE", "}"),
    ]


def test_whitespace_is_ignored():
    """
    验证空格、换行、Tab 不会产生额外 Token。
    """

    source1 = "pipeline test {}"
    source2 = "pipeline    test   {\n\t}"

    assert tokens_of(source1) == tokens_of(source2)


def test_illegal_character():
    """
    @ 不属于当前 FlowGPU 词法，因此应该触发词法错误。
    """

    with pytest.raises(UnexpectedCharacters, match='No terminal matches'):
        tokens_of("pipeline test @")


def test_no_token_is_declared_without_a_rule_behind_it():
    """The lexer must not know a token the parser can never reach.

    The two views of the single grammar are built from the same file, but
    they do not agree by construction: the standalone lexer knows every
    *declared* terminal, while the parser only offers the terminals that are
    *reachable* in some state. DIV was once declared with no rule using it,
    and that is exactly how they disagreed -- the lexer produced a DIV token
    while the parser answered "no terminal matches '/'".
    """
    from flowgpu.compiler.lexer import lexer
    from flowgpu.compiler.parser import parser

    declared = {token.name for token in lexer.terminals}
    reachable = {token.name for token in parser.terminals}

    assert declared == reachable


def test_division_has_no_token():
    """v0.1 has no `div` type rule, so it has no DIV terminal either.

    Registering the operator comes first (operator.py), then the token.
    """
    with pytest.raises(UnexpectedCharacters, match='No terminal matches'):
        tokens_of("/")


def test_every_infix_token_desugars_to_a_registered_operator():
    """The chain is token -> INFIX_OPERATORS -> operator registry.

    Every link has to exist, and the two ends are the ones that get
    forgotten: DIV once sat at the head of this chain with nothing behind
    it -- a token no rule could reach and no operator could type.
    """
    from flowgpu.compiler.ast_builder import INFIX_OPERATORS
    from flowgpu.compiler.lexer import lexer
    from flowgpu.compiler.operator import OPERATORS

    signs = {"+", "-", "*", "/", "%"}
    infix_tokens = {
        token.name
        for token in lexer.terminals
        if str(token.pattern.value) in signs
    }

    assert infix_tokens == set(INFIX_OPERATORS)
    assert set(INFIX_OPERATORS.values()) <= set(OPERATORS)