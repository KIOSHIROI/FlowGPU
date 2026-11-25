"""Lesson 3 fuses reduction, broadcast and multiply."""
import json
from pathlib import Path

from flowgpu.compiler.compile import compile_source

ROOT = Path("lessons/03-reduction")


def test_reduction_stays_in_one_primitive_kernel():
    result = compile_source((ROOT / "lesson.fgpu").read_text(),
                            shape_bindings={"N": 4, "D": 8}).pipelines[0]
    assert len(result.triton.kernels) == 1
    kernel = result.triton.kernels[0]
    assert kernel.kind == "primitive"
    assert kernel.ops == ("sum_last", "broadcast_like", "mul")


def test_reduction_quiz_is_complete():
    quiz = json.loads((ROOT / "quiz.json").read_text())
    assert len(quiz["questions"]) == 5
    assert all(item["answer"] for item in quiz["questions"])
