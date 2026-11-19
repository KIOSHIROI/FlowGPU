"""Structured diagnostics -- Design.md 17.3, made checkable.

That section promises the compiler can return TypeError, ShapeError,
OperatorError and DependencyError. Before this file existed, four of those
arrived as one "type" phase with no position, because the category was
decided from the Python exception class:

    except ValueError -> name error
    except TypeError  -> type error

which cannot tell a dtype mismatch from a shape mismatch, and says nothing
about where the problem is. The category is now decided where it is known:
in the operator's own rule (operator.py) or in the symbol table, and the
pass that holds the AST node attaches its position.
"""

import json

import pytest

from compiler.compile import CompileError, compile_source
from compiler.diagnostic import DiagnosticKind


def diagnostic(source: str) -> CompileError:
    with pytest.raises(CompileError, match=', column') as caught:
        compile_source(source)

    return caught.value


# --- DependencyError: a dependency with no producer, or two of them ------


def test_undefined_symbol_points_at_the_undefined_name():
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.DEPENDENCY
    assert "Undefined symbol 'r'" == error.message
    # Not the `add` and not the statement: the name that has no producer.
    assert (error.line, error.column) == (3, 16)


def test_forward_reference_is_a_dependency_error():
    """def-before-use is a dataflow rule, so breaking it is a dependency."""
    error = diagnostic(
        "pipeline p {\n"
        "    y = add(x, r)\n"
        "    input x : bf16[N,D]\n"
        "    input r : bf16[N,D]\n"
        "    output y\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.DEPENDENCY
    assert "Undefined symbol 'x'" == error.message
    assert (error.line, error.column) == (2, 13)


def test_duplicate_definition_is_a_dependency_error():
    """One name, one value: a second producer makes the def-use chain
    ambiguous, which is the same defect as having none."""
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input x : fp16[N,D]\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.DEPENDENCY
    assert "Symbol 'x' is already defined" == error.message
    assert (error.line, error.column) == (3, 11)


# --- TypeError and ShapeError: two different things, two different kinds --


def test_dtype_mismatch_is_a_type_error():
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input r : fp16[N,D]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.TYPE
    assert "Cannot add bf16 and fp16" == error.message
    assert (error.line, error.column) == (4, 9)


def test_shape_mismatch_is_a_shape_error():
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input r : bf16[N,H]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.SHAPE
    assert "shapes" in error.message
    assert (error.line, error.column) == (4, 9)


def test_the_position_of_an_operator_is_the_operator():
    """`x + r` points at the `+`, and `add(x, r)` at the `add`.

    Both spellings reach the same node (it is sugar for the operator call),
    so a diagnostic points at the operator the user wrote -- not at the
    statement, and not at one of the operands.
    """
    infix = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input r : fp16[N,D]\n"
        "    y = x + r\n"
        "    output y\n"
        "}\n"
    )
    written_as_a_call = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input r : fp16[N,D]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )

    assert (infix.line, infix.column) == (4, 11)
    assert (written_as_a_call.line, written_as_a_call.column) == (4, 9)
    assert infix.message == written_as_a_call.message


def test_a_rank_restriction_is_a_shape_error():
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[B,S,D]\n"
        "    y = transpose(x)\n"
        "    output y\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.SHAPE
    assert "2D" in error.message
    assert (error.line, error.column) == (3, 9)


def test_a_message_stays_inside_the_language():
    """`[N,D]`, not Python's `['N', 'D']`.

    Diagnostic text is read by a repair loop, and whatever it sees should be
    something the user could have written in the source.
    """
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input r : bf16[N,H]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )

    assert error.message == "Cannot add tensors with shapes [N,D] and [N,H]"


def test_the_two_kinds_are_distinguishable_from_the_same_operator():
    """Same operator, same line, two different categories.

    This is the pair the old design could not tell apart: both were a
    "type" error, so a repair loop could not know whether to change a dtype
    or a dimension.
    """
    dtype_error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[2,4]\n"
        "    input r : fp16[4,2]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )
    shape_error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[2,4]\n"
        "    input r : bf16[4,2]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )

    assert dtype_error.kind is DiagnosticKind.TYPE
    assert shape_error.kind is DiagnosticKind.SHAPE


# --- OperatorError: the operator, not its operands, is at fault ----------


def test_unknown_operator_is_an_operator_error():
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    y = foo(x)\n"
        "    output y\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.OPERATOR
    assert "Unknown operator 'foo'" == error.message
    assert (error.line, error.column) == (3, 9)


def test_wrong_argument_count_is_an_operator_error():
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    y = rms_norm(x)\n"
        "    output y\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.OPERATOR
    assert "expects 2 arguments, got 1" in error.message
    assert (error.line, error.column) == (3, 9)


def test_a_call_with_no_arguments_counts_zero():
    """Regression: `rms_norm()` used to be read as one argument.

    The grammar always yields the parenthesis tokens, so the "no arguments"
    branch never fired and the RPAR string was taken as a single argument --
    which made the arity diagnostic report "got 1" for a call that passed
    nothing.
    """
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    y = rms_norm()\n"
        "    output y\n"
        "}\n"
    )

    assert "got 0" in error.message


def test_a_multi_value_operator_under_one_name_is_an_operator_error():
    """v0.2 replaced the reserved-operator error with a real one: `split` is
    implemented, and the mistake is now the call site -- one name cannot
    select a value from an operator that produces several."""
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    y = split(x)\n"
        "    output y\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.OPERATOR
    assert "several values" in error.message
    assert (error.line, error.column) == (3, 9)


def test_a_multi_name_assignment_of_a_single_value_operator_is_rejected():
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    a, b = relu(x)\n"
        "    output a\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.OPERATOR
    assert "produces one value" in error.message


# --- SyntaxError: the fifth kind, for a file that does not parse ----------


def test_a_syntax_error_carries_a_position():
    error = diagnostic(
        "pipeline p {\n"
        "    input x bf16[N,D]\n"
        "    output x\n"
        "}\n"
    )

    assert error.kind is DiagnosticKind.SYNTAX
    assert (error.line, error.column) == (2, 13)


def test_a_lexical_error_carries_a_position():
    error = diagnostic("pipeline p { input x : bf16[N,D] @ }\n")

    assert error.kind is DiagnosticKind.SYNTAX
    assert (error.line, error.column) == (1, 34)


# --- The promises themselves --------------------------------------------


SOURCES_PER_KIND = {
    DiagnosticKind.DEPENDENCY: (
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input x : bf16[N,D]\n"
        "}\n"
    ),
    DiagnosticKind.TYPE: (
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input r : fp16[N,D]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    ),
    DiagnosticKind.SHAPE: (
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input r : bf16[N,H]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    ),
    DiagnosticKind.OPERATOR: (
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    y = split(x)\n"
        "    output y\n"
        "}\n"
    ),
    # Still an OPERATOR error, but for a v0.2 reason (a multi-value operator
    # under one name) rather than v0.1's "reserved" refusal.
}


def test_design_md_17_3_semantic_kinds_are_all_reachable():
    """The four names in that section, each produced by a real program.

    A promise in a design document is worth nothing until something fails
    when it stops being true.
    """
    assert set(SOURCES_PER_KIND) == {
        DiagnosticKind.DEPENDENCY,
        DiagnosticKind.TYPE,
        DiagnosticKind.SHAPE,
        DiagnosticKind.OPERATOR,
    }

    for expected, source in SOURCES_PER_KIND.items():
        assert diagnostic(source).kind is expected


def test_every_semantic_diagnostic_carries_a_position():
    """The gap this step closes: semantic errors used to have no location."""
    for kind, source in SOURCES_PER_KIND.items():
        error = diagnostic(source)

        assert error.kind is kind
        assert error.position is not None, kind
        assert error.line is not None and error.column is not None, kind
        assert error.line >= 1 and error.column >= 1, kind


# --- The shape a repair loop consumes -----------------------------------


def test_a_diagnostic_serialises_to_json():
    error = diagnostic(
        "pipeline p {\n"
        "    input x : bf16[N,D]\n"
        "    input r : fp16[N,D]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )

    payload = json.loads(json.dumps(error.as_dict()))

    assert payload == {
        "pipeline": "p",
        "kind": "type",
        "message": "Cannot add bf16 and fp16",
        "line": 4,
        "column": 9,
    }


def test_the_message_does_not_repeat_the_position():
    """The position is a field, so it must not also be inside the text.

    Lark writes "at line N, column M" (or "at line N col M") into its
    message, and it used to reach the user twice.
    """
    error = diagnostic("pipeline p {\n    input x bf16[N,D]\n}\n")

    assert "at line" not in error.message
    assert "col" not in error.message


def test_the_message_does_not_keep_the_comma_before_the_position():
    """A lexical error is worded "..., at line 1 col 34": stripping the
    location must not leave the comma stranded."""
    error = diagnostic("pipeline p { input x : bf16[N,D] @ }\n")

    assert error.message == (
        "No terminal matches '@' in the current parser context"
    )


def test_the_pipeline_name_is_reported():
    source = (
        "pipeline a { input x : bf16[N,D] output x }\n"
        "pipeline b {\n"
        "    input x : bf16[N,D]\n"
        "    input r : fp16[N,D]\n"
        "    y = add(x, r)\n"
        "    output y\n"
        "}\n"
    )

    error = diagnostic(source)

    assert error.pipeline == "b"
    assert error.as_dict()["pipeline"] == "b"
