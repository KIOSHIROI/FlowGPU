"""The shipped examples must compile -- and the broken ones must not."""

from pathlib import Path

import pytest

from compiler.compile import CompileError, compile_source
from compiler.diagnostic import DiagnosticKind

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"

GOOD = sorted(EXAMPLES.glob("*.fgpu"))
BAD = sorted((EXAMPLES / "errors").glob("*.fgpu"))

# What each broken example is broken *of*. Naming it here rather than
# accepting any failure means a file that starts failing for an unrelated
# reason is a test failure, not a pass.
EXPECTED_KIND = {
    "dtype_mismatch.fgpu": DiagnosticKind.TYPE,
    "shape_mismatch.fgpu": DiagnosticKind.SHAPE,
}


def test_both_kinds_of_example_are_present():
    assert GOOD, "no examples found"
    assert BAD, "no error examples found"


def test_every_error_example_declares_the_kind_it_should_produce():
    """Adding a broken example means deciding what is broken about it."""
    assert set(EXPECTED_KIND) == {path.name for path in BAD}


@pytest.mark.parametrize("path", GOOD, ids=lambda p: p.name)
def test_example_compiles(path):
    source = path.read_text(encoding="utf-8")
    result = compile_source(source, source_path=str(path))

    assert len(result.pipelines) == 1
    compiled = result.pipelines[0]
    assert compiled.pipeline.name == path.stem

    # Every value and output must have come out typed.
    for name, symbol in compiled.symbols.symbols.items():
        assert symbol.type is not None, f"{name} has no type"


@pytest.mark.parametrize("path", BAD, ids=lambda p: p.name)
def test_broken_example_is_rejected(path):
    source = path.read_text(encoding="utf-8")

    with pytest.raises(CompileError, match='pe error in pipeline') as caught:
        compile_source(source, source_path=str(path))

    assert caught.value.kind is EXPECTED_KIND[path.name]
    assert caught.value.line is not None
