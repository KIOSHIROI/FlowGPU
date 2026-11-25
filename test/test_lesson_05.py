"""Lesson 5 fuses the SwiGLU epilogue into the up-projection."""
import json
from pathlib import Path

from flowgpu.compiler.compile import compile_source

ROOT = Path("lessons/05-fusion")


def test_swiglu_fuses_up_and_act():
    result = compile_source((ROOT / "lesson.fgpu").read_text(),
                            shape_bindings={"B": 2, "D": 8, "H": 16}).pipelines[0]
    assert len(result.triton.kernels) == 3
    kernel = result.triton.kernels[1]
    assert kernel.kind == "matmul"
    assert kernel.ops == ("matmul", "silu", "mul")


def test_fusion_quiz_is_complete():
    quiz = json.loads((ROOT / "quiz.json").read_text())
    assert len(quiz["questions"]) == 5
    assert all(item["answer"] for item in quiz["questions"])
