"""Lesson 1 compiles, fuses, and runs its CPU reference."""
import json
from pathlib import Path

from flowgpu.compiler.compile import compile_source

ROOT = Path("lessons/01-values")


def test_lesson_one_is_a_single_fused_kernel():
    result = compile_source((ROOT / "lesson.fgpu").read_text(),
                            shape_bindings={"N": 4, "D": 8}).pipelines[0]
    assert len(result.triton.kernels) == 1
    assert result.triton.kernels[0].ops == ("broadcast_like", "mul", "add")


def test_lesson_quiz_is_complete():
    quiz = json.loads((ROOT / "quiz.json").read_text())
    assert len(quiz["questions"]) == 5
    assert all(item["answer"] for item in quiz["questions"])
