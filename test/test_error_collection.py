"""17.3's repair loop should hear about every mistake, not the first one.

A pass that raises on the first bad statement costs a compile per mistake. These
tests hold the two halves of the improvement -- mistakes that are genuinely
independent are all reported, and a diagnostic that is a *consequence* of an
earlier one is reported too but with the note that says so out loud.

That note is the honest limit. A statement that failed never defined its name, so
the statements that read that name fail as well, and telling the two apart needs
a name that can be marked poisoned -- a type the rules pass through without
complaining. The language has no such type yet, so the cascade is reported rather
than hidden. Marking it poisoned is the named next step, not a thing this pass
quietly gets away with.
"""

import json

from compiler.cli import main
from compiler.compile import compile_source
from compiler.diagnostic import CompileError

# Two mistakes that do not depend on each other: both assignments are wrong, and
# neither name is read afterwards, so neither failure can produce the other.
TWO_INDEPENDENT = """pipeline p {
    input x : bf16[N,D]
    input r : fp16[N,D]
    input s : fp32[N,D]
    a = add(x, r)
    b = add(x, s)
}
"""

TWO_PIPELINES = """pipeline a {
    input x : bf16[N,D]
    input r : fp16[N,D]
    z = add(x, r)
}
pipeline b {
    input u : bf16[N,D]
    w = sub(u, u, u)
}
"""


def errors_of(source):
    try:
        compile_source(source)
    except CompileError as error:
        return error.all_diagnostics()

    raise AssertionError("this program was supposed to fail")


def test_independent_mistakes_are_all_reported():
    found = errors_of(TWO_INDEPENDENT)

    assert [error.message for error in found] == [
        "Cannot add bf16 and fp16",
        "Cannot add bf16 and fp32",
    ]
    assert [error.line for error in found] == [5, 6]
    assert found[0].pipeline == found[1].pipeline == "p"
    assert all(error.kind.value == "type" for error in found)


def test_every_failing_pipeline_is_reported():
    found = errors_of(TWO_PIPELINES)

    assert [error.pipeline for error in found] == ["a", "b"]
    assert found[1].line == 8


def test_a_parse_error_is_still_one_error():
    """There is nothing to walk when the file does not parse.

    Collecting semantic errors is a property of a pass that walks statements;
    it cannot make a syntax error into several, and it should not pretend to.
    """
    found = errors_of("pipeline p {\n    input x : bf16[N,D] @\n}\n")

    assert len(found) == 1
    assert found[0].kind.value == "syntax"


CASCADE_PROGRAM = """pipeline p {
    input x : bf16[N,D]
    input r : fp16[N,D]
    y = add(x, r)
    z = silu(y)
    output z
}
"""


def test_cascading_diagnostics_are_suppressed():
    """When `y` fails to typecheck, neither `z = silu(y)` nor `output z` cascades."""
    found = errors_of(CASCADE_PROGRAM)

    assert len(found) == 1
    assert found[0].message == "Cannot add bf16 and fp16"
    assert found[0].line == 4


def test_the_command_line_prints_independent_diagnostics_without_noise(tmp_path, capsys):
    path = tmp_path / "two.fgpu"
    path.write_text(TWO_INDEPENDENT, encoding="utf-8")

    assert main([str(path)]) == 1

    printed = capsys.readouterr().err
    lines = printed.splitlines()

    assert len(lines) == 2
    assert lines[0].startswith("error: type error in pipeline 'p' at line 5")
    assert lines[1].startswith("error: type error in pipeline 'p' at line 6")
    assert not any("note:" in line for line in lines)


def test_one_mistake_prints_exactly_one_line(tmp_path, capsys):
    """The common case must not grow a note about a cascade that is not there."""
    path = tmp_path / "one.fgpu"
    path.write_text(
        "pipeline p {\n    input x : bf16[N,D] @\n}\n", encoding="utf-8"
    )

    assert main([str(path)]) == 1

    printed = capsys.readouterr().err

    assert printed.count("\n") == 1
    assert "note:" not in printed


def test_the_json_form_is_an_array_when_there_is_more_than_one(tmp_path, capsys):
    path = tmp_path / "two.fgpu"
    path.write_text(TWO_INDEPENDENT, encoding="utf-8")

    assert main([str(path), "--emit", "json"]) == 1

    payload = json.loads(capsys.readouterr().err)

    assert isinstance(payload, list)
    assert [diagnostic["line"] for diagnostic in payload] == [5, 6]
    assert all(diagnostic["pipeline"] == "p" for diagnostic in payload)
