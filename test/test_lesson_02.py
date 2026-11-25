"""Lesson 2 keeps split/interleave/concat in one primitive kernel."""
import json
from pathlib import Path

from flowgpu.compiler.compile import compile_source

ROOT = Path("lessons/02-layout")


def test_layout_ops_stay_in_one_primitive_kernel():
    result = compile_source((ROOT / "lesson.fgpu").read_text(),
                            shape_bindings={"N": 4}).pipelines[0]
    assert len(result.triton.kernels) == 1
    kernel = result.triton.kernels[0]
    assert kernel.kind == "primitive"
    assert kernel.ops == ("split", "split", "interleave", "concat", "add")


def test_layout_quiz_is_complete():
    quiz = json.loads((ROOT / "quiz.json").read_text())
    assert len(quiz["questions"]) == 6
    assert all(item["answer"] for item in quiz["questions"])
