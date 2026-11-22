"""Selecting one pipeline from a multi-pipeline source before preparation."""
from pathlib import Path

import pytest

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.runtime.preparation import select_pipeline_source

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "examples" / "batch8" / "wide_row_softmax.fgpu").read_text()


def test_selects_only_the_named_pipeline():
    selected = select_pipeline_source(SOURCE, "wide_plain")
    assert "pipeline wide_plain" in selected
    assert "pipeline wide_temp" not in selected
    compiled = compile_source(selected)
    assert [p.pipeline.name for p in compiled.pipelines] == ["wide_plain"]


def test_unknown_pipeline_is_an_error():
    with pytest.raises(ValueError, match="pipeline 'missing' not found"):
        select_pipeline_source(SOURCE, "missing")
