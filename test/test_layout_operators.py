"""The v0.2 language layer for reshape/split/concat.

These tests hold what the language now accepts and what it rejects --
grammar, multi-assignment binding, and the type rules. The backends refuse
the three operators until host lowering lands (same shape of refusal as
attention on the pure-Triton target), and that refusal is pinned here too,
because "accepted by the language, refused by the target" is a contract,
not an accident.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compiler.ast_node import Assignment, ShapeLiteral
from compiler.compile import compile_source
from compiler.diagnostic import CompileError, DiagnosticKind
from compiler.parser import parser
from compiler.ast_builder import ASTBuilder


SPLIT3 = """
pipeline qkv {
    input x : bf16[N,3072]
    q, k, v = split(x)
    output q
}
"""


def parse_statement(source):
    tree = parser.parse(source)
    return ASTBuilder().transform(tree)[0].statements


# ---- the grammar accepts what v0.1 reserved ----

def test_multi_assignment_parses_with_ordered_names():
    statement = parse_statement(
        "pipeline p { input x : bf16[N,8] a, b, c = split(x) output a }"
    )[1]
    assert isinstance(statement, Assignment)
    assert statement.names == ["a", "b", "c"]
    assert statement.value.name == "split"


def test_a_single_name_assignment_keeps_its_one_name():
    statement = parse_statement(
        "pipeline p { input x : bf16[N,8] y = relu(x) output y }"
    )[1]
    assert statement.names == ["y"]
    assert statement.name == "y"


def test_multi_name_reads_raise_rather_than_guess():
    statement = parse_statement(
        "pipeline p { input x : bf16[N,8] a, b = split(x) output a }"
    )[1]
    with pytest.raises(AttributeError, match='multi-name assignment a, b has no single \\.name'):
        statement.name  # noqa: B018 -- the property must raise, not return a


def test_a_shape_literal_parses_as_its_own_node():
    statement = parse_statement(
        "pipeline p { input x : bf16[N,2048] y = reshape(x, [N,32,64]) output y }"
    )[1]
    assert isinstance(statement.value.args[1], ShapeLiteral)
    assert statement.value.args[1].shape == ["N", "32", "64"]


# ---- the type rules bind one type per name ----

def test_split_binds_each_name_with_the_part_shape():
    result = compile_source(SPLIT3)
    symbols = result.pipelines[0].symbols
    assert symbols.lookup("q").type.shape == ["N", "1024"]
    assert symbols.lookup("k").type.shape == ["N", "1024"]
    assert symbols.lookup("v").type.shape == ["N", "1024"]
    assert symbols.lookup("q").type.dtype == "bf16"


def test_two_way_split_of_a_symbolic_leading_axis():
    result = compile_source("""
    pipeline halves {
        input x : bf16[B, 64]
        left, right = split(x)
        output left
    }
    """)
    assert result.pipelines[0].symbols.lookup("left").type.shape == ["B", "32"]


def test_reshape_retypes_to_the_literal_shape():
    result = compile_source("""
    pipeline rs {
        input x : bf16[N, 2048]
        y = reshape(x, [N, 32, 64])
        output y
    }
    """)
    assert result.pipelines[0].symbols.lookup("y").type.shape == ["N", "32", "64"]


def test_reshape_permutes_symbolic_dims_when_counts_hold():
    """[N, 2048] -> [2048, N] is a legal row-major reinterpret: 2048*N
    elements either way, and the checker can prove it."""
    result = compile_source("""
    pipeline rs {
        input x : bf16[N, 2048]
        y = reshape(x, [2048, N])
        output y
    }
    """)
    assert result.pipelines[0].symbols.lookup("y").type.shape == ["2048", "N"]


def test_concat_sums_the_literal_last_axes():
    result = compile_source("""
    pipeline cc {
        input a : bf16[N, 64]
        input b : bf16[N, 32]
        y = concat(a, b)
        output y
    }
    """)
    assert result.pipelines[0].symbols.lookup("y").type.shape == ["N", "96"]


# ---- the rejections are each their own reason ----

def test_a_symbolic_last_axis_cannot_be_split():
    with pytest.raises(CompileError, match='olumn 20: split needs a literal last axis, got bf16\\[N,D\\]: the part width cannot be written from a symbolic dimension') as caught:
        compile_source("""
        pipeline s {
            input x : bf16[N, D]
            a, b = split(x)
            output a
        }
        """, layout_views=False)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "literal last axis" in caught.value.message


def test_an_indivisible_split_is_rejected():
    with pytest.raises(CompileError, match='olumn 23: split cannot cut a last axis of 2050 into 3 equal parts') as caught:
        compile_source("""
        pipeline s {
            input x : bf16[N, 2050]
            a, b, c = split(x)
            output a
        }
        """)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "2050" in caught.value.message


def test_reshape_refuses_a_count_change():
    with pytest.raises(CompileError, match='olumn 17: reshape cannot turn bf16\\[2,4\\] into \\[2,5\\]: 8 elements cannot become 10') as caught:
        compile_source("""
        pipeline s {
            input x : bf16[2, 4]
            y = reshape(x, [2, 5])
            output y
        }
        """)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "8 elements" in caught.value.message


def test_reshape_refuses_the_minus_one_placeholder():
    """The placeholder cannot even be written: `dimension` is NAME or INT,
    so -1 fails at parse. The grammar refuses what the rule would have to
    guess."""
    from lark.exceptions import LarkError

    with pytest.raises(CompileError, match='olumn 32: Unexpected token Token') as caught:
        compile_source("""
        pipeline s {
            input x : bf16[2, 4]
            y = reshape(x, [2, -1])
            output y
        }
        """)
    assert caught.value.kind is DiagnosticKind.SYNTAX


def test_concat_refuses_a_symbolic_last_axis():
    """The joined width 64+D is not writable: no input declares it, no host
    could allocate it. Symmetric with split's literal-last-axis rule."""
    with pytest.raises(CompileError, match='olumn 17: concat requires literal last axes: the joined width of bf16\\[N,64\\] and bf16\\[N,D\\] cannot be written from a symbolic dimension') as caught:
        compile_source("""
        pipeline s {
            input a : bf16[N, 64]
            input b : bf16[N, D]
            y = concat(a, b)
            output y
        }
        """)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "literal last axes" in caught.value.message


def test_reshape_refuses_a_dimension_no_input_declares():
    """A symbol in the target shape is a name, and names are bound by input
    declarations; the host reads every symbol off an input tensor's shape,
    so an undeclared one has no run-time value to allocate with."""
    with pytest.raises(CompileError, match="olumn 17: reshape dimension 'M' is not declared by any input; a symbol is bound by the input tensor that carries it") as caught:
        compile_source("""
        pipeline s {
            input x : bf16[2, 4]
            y = reshape(x, [M, 8])
            output y
        }
        """)
    assert caught.value.kind is DiagnosticKind.DEPENDENCY
    assert "M" in caught.value.message


def test_concat_refuses_mismatched_leading_axes():
    with pytest.raises(CompileError, match='olumn 17: concat requires matching leading dimensions: bf16\\[N,64\\] and bf16\\[M,32\\]') as caught:
        compile_source("""
        pipeline s {
            input a : bf16[N, 64]
            input b : bf16[M, 32]
            y = concat(a, b)
            output y
        }
        """)
    assert caught.value.kind is DiagnosticKind.SHAPE
    assert "leading dimensions" in caught.value.message


def test_a_shape_literal_outside_reshape_is_a_type_error():
    with pytest.raises(CompileError, match="olumn 24: a shape literal cannot be an argument of 'add'; it only names reshape's target shape") as caught:
        compile_source("""
        pipeline s {
            input a : bf16[2, 4]
            input b : bf16[2, 4]
            y = add(a, [2, 4])
            output y
        }
        """)
    assert caught.value.kind is DiagnosticKind.TYPE
    assert "shape literal" in caught.value.message


def test_a_bare_shape_literal_is_not_a_value():
    with pytest.raises(CompileError, match="olumn 17: a shape literal only names reshape's target shape; it is not a tensor") as caught:
        compile_source("""
        pipeline s {
            input a : bf16[2, 4]
            y = [2, 4]
            output y
        }
        """)
    assert caught.value.kind is DiagnosticKind.TYPE


# ---- the graph: one node per value, part-indexed ----

def test_split_lowers_to_one_node_per_name():
    result = compile_source(SPLIT3)
    graph = result.pipelines[0].dataflow
    splits = [node for node in graph.nodes if node.op == "split"]
    assert len(splits) == 3
    assert sorted(node.part for node in splits) == [0, 1, 2]
    assert {node.output for node in splits} == {"q", "k", "v"}
    # Every value still has exactly one producer.
    for value in ("q", "k", "v"):
        assert graph.producer_of(value).op == "split"


def test_a_reshape_node_has_no_shape_input():
    """The shape literal is metadata for the rule, never an operand."""
    result = compile_source("""
    pipeline rs {
        input x : bf16[N, 2048]
        y = reshape(x, [N, 32, 64])
        output y
    }
    """)
    node = result.pipelines[0].dataflow.producer_of("y")
    assert node.op == "reshape"
    assert node.inputs == ["x"]


# ---- both backends lower them as host-side layout kernels ----

def test_the_backends_lower_split_as_host_kernels():
    # All three parts are read here; a program whose part is dead gets it
    # skipped by the buffer plan, which is the point of the dead-value test
    # below, not an accident this test should depend on.
    program = compile_source("""
    pipeline qkv {
        input x : bf16[N, 3072]
        q, k, v = split(x)
        y = concat(concat(q, k), v)
        output y
    }
    """).pipelines[0]
    triton = program.triton
    assert triton.complete
    assert {kernel.kind for kernel in triton.kernels} == {"layout"}
    # The three parts slice the input's last axis at literal bounds.
    sources = " ".join(kernel.source for kernel in triton.kernels)
    assert "0:1024" in sources and "1024:2048" in sources and "2048:3072" in sources
    # No device kernel: the call is a plain name, no [grid] subscript.
    for line in triton.host.splitlines():
        if "_k" in line and "(" in line:
            assert "[" not in line
    reference = program.reference

    assert reference.complete
    assert {kernel.kind for kernel in reference.kernels} == {"layout"}


def test_a_layout_round_trip_restores_the_input():
    """concat(a, b) is the inverse of split, and a reshape to the input's
    own shape is the identity -- so the composition of all three operators
    must leave every element where it was. That is checked against the
    reference backend, which runs: the program is executed, not just
    emitted."""
    result = compile_source("""
    pipeline roundtrip {
        input x : bf16[N, 64]
        a, b = split(x)
        y = concat(a, b)
        z = reshape(y, [N, 64])
        output z
    }
    """)
    program = result.pipelines[0]
    from compiler import reference as reference_backend

    namespace = {}
    exec(compile(reference_backend.module([program.reference]), "<reference>", "exec"), namespace)
    data = [float(index % 31) - 10.0 for index in range(4 * 64)]
    out = namespace["run_roundtrip"](data, 4)
    assert out["z"] == data


def test_the_reshape_runtime_guard_fires_at_run_time():
    """The symbolic conservation the type layer cannot prove is checked by
    the generated code, with the value in hand: a caller whose N makes the
    counts disagree gets the error, not silent garbage."""
    from compiler import reference as reference_backend

    program = compile_source("""
    pipeline wrong {
        input x : bf16[N, 2048]
        y = reshape(x, [N, 1024])
        output y
    }
    """).pipelines[0]
    namespace = {}
    exec(compile(reference_backend.module([program.reference]), "<reference>", "exec"), namespace)
    data = [0.5] * (4 * 2048)
    with pytest.raises(ValueError, match="does not conserve elements"):
        namespace["run_wrong"](data, 4)


def test_the_triton_reshape_guard_is_reachable_code():
    """The guard is not a comment: parsed as Python, the raise must sit
    under an if whose test is not a constant. A guard folded to `False and
    ...` still contains the message and passes any text search."""
    import ast as python_ast

    program = compile_source("""
    pipeline wrong {
        input x : bf16[N, 2048]
        y = reshape(x, [N, 1024])
        output y
    }
    """).pipelines[0]
    kernel = next(k for k in program.triton.kernels if "reshape" in k.source)
    tree = python_ast.parse(kernel.source)
    raises = [
        node for node in python_ast.walk(tree)
        if isinstance(node, python_ast.Raise)
    ]
    assert raises, "the guard's raise is gone"
    for node in raises:
        test = getattr(node, "reachable", None)  # placeholder; real check below
    # The enclosing if must not be statically false.
    conditions = [
        n.test for n in python_ast.walk(tree)
        if isinstance(n, python_ast.If)
        and any(isinstance(x, python_ast.Raise) for x in python_ast.walk(n))
    ]
    assert conditions, "the raise is not guarded by an if"
    for test in conditions:
        assert not (
            isinstance(test, python_ast.BoolOp)
            and isinstance(test.values[0], python_ast.Constant)
            and test.values[0].value is False
        ), "the guard was folded to a constant"


def test_concat_lowers_along_the_last_axis():
    """The language defines concat on the last axis (the axis parameter is
    unwritable -- that was v0.1's refusal), so the emitted call must name
    exactly that axis."""
    program = compile_source("""
    pipeline p {
        input a : bf16[N, 64]
        input b : bf16[N, 32]
        y = concat(a, b)
        output y
    }
    """).pipelines[0]
    kernel = next(k for k in program.triton.kernels if "torch.cat" in k.source)
    assert "dim=-1" in kernel.source


def test_pointwise_work_does_not_fuse_into_a_layout_op():
    """A split has no registers to host a relu in: the copy is host code,
    so an epilogue on it would be a category error. relu keeps its own
    pointwise kernel, and the split group stays standalone."""
    program = compile_source("""
    pipeline qkv {
        input x : bf16[N, 3072]
        q, k, v = split(x)
        h = relu(q)
        output h
    }
    """, layout_views=False).pipelines[0]
    for group in program.fusion.groups:
        if group.anchor is not None and group.anchor.op == "split":
            assert group.epilogue == () and group.prologue == ()
            assert [node.op for node in group.members] == ["split"]
    kinds = [
        kernel.kind for kernel in program.triton.kernels
        if "relu" in kernel.source
    ]
    assert kinds == ["pointwise"]


def test_the_reshape_runtime_guard_is_emitted_by_both_backends():
    program = compile_source("""
    pipeline wrong {
        input x : bf16[N, 2048]
        y = reshape(x, [N, 1024])
        output y
    }
    """).pipelines[0]
    # The guard lives in the emitted function on both targets, and the
    # Triton host calls that function, so the check runs wherever the
    # program runs.
    assert "does not conserve elements" in "".join(
        kernel.source for kernel in program.triton.kernels
    )
    assert "does not conserve elements" in "".join(
        kernel.source for kernel in program.reference.kernels
    )
