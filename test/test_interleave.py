"""The `interleave` layout operator: out[..., 2i] = a[..., i]."""
from __future__ import annotations

import pytest

from compiler import compile_source
from compiler.backend import reference
from compiler.backend.triton_backend import module as triton_module
from compiler.diagnostic import CompileError

PROGRAM = """
pipeline il {
  input a : int32[2,3]
  input b : int32[2,3]
  c = interleave(a, b)
  output c
}
"""


def _compile(source=PROGRAM):
    return compile_source(source).pipelines[0]


def test_interleave_doubles_the_last_axis_and_alternates_the_operands():
    pipeline = _compile()
    assert pipeline.triton.complete and pipeline.reference.complete
    namespace = {}
    exec(reference.module([pipeline.reference]), namespace)
    produced = namespace["run_il"]([1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60])["c"]
    assert produced == [1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 6, 60]


def test_interleave_lowers_to_the_host_layout_call():
    text = triton_module([_compile().triton])
    assert "torch.stack" in text
    assert "copy_" in text


@pytest.mark.parametrize("source, message", [
    ("pipeline p { input a : int32[2,3] input b : int32[2,4] c = interleave(a, b) output c }",
     "matching shapes"),
    ("pipeline p { input a : int32[2,3] input b : fp16[2,3] c = interleave(a, b) output c }",
     "matching dtypes"),
    ("pipeline p { input a : int32[3,N] input b : int32[3,N] c = interleave(a, b) output c }",
     "literal last axis"),
])
def test_interleave_requires_matching_literal_last_axis_operands(source, message):
    with pytest.raises(CompileError, match=message):
        _compile(source)
