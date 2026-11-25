"""Lesson 7 shows the generated tail masks."""
import json
from pathlib import Path

from flowgpu.compiler.backend.triton_backend import module
from flowgpu.compiler.compile import compile_source

ROOT = Path("lessons/07-masks")


def test_generated_code_has_all_three_masks():
    program = compile_source((ROOT / "lesson.fgpu").read_text(),
                             shape_bindings={"M": 3, "K": 7, "N": 5}).pipelines[0]
    code = module([program.triton])
    assert "offs_m[:, None] < 3" in code
    assert "offs_n[None, :] < 5" in code
    assert "depth_offsets[None, :] < 7" in code
    assert "other=0.0" in code


def test_mask_quiz_is_complete():
    quiz = json.loads((ROOT / "quiz.json").read_text())
    assert len(quiz["questions"]) == 5
    assert all(item["answer"] for item in quiz["questions"])
