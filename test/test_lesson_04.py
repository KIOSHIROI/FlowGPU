"""Lesson 4 compiles to one matmul kernel."""
import json
from pathlib import Path

from flowgpu.compiler.compile import compile_source

ROOT = Path("lessons/04-matmul")


def test_matmul_is_one_kernel():
    result = compile_source((ROOT / "lesson.fgpu").read_text(),
                            shape_bindings={"M": 4, "K": 8, "N": 16}).pipelines[0]
    assert len(result.triton.kernels) == 1
    assert result.triton.kernels[0].kind == "matmul"


def test_matmul_quiz_is_complete():
    quiz = json.loads((ROOT / "quiz.json").read_text())
    assert len(quiz["questions"]) == 5
    assert all(item["answer"] for item in quiz["questions"])
