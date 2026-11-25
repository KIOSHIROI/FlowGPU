"""Lesson 8 compares one fused kernel with a two-kernel handwritten version."""
import json
from pathlib import Path

from flowgpu.compiler.compile import compile_source

ROOT = Path("lessons/08-handwritten")
PACKET = Path("review/swiglu_split")


def test_generated_form_is_a_single_primitive_kernel():
    program = compile_source((PACKET / "source.fgpu").read_text()).pipelines[0]
    assert len(program.triton.kernels) == 1
    assert program.triton.kernels[0].kind == "primitive"


def test_handwritten_form_has_two_kernels():
    text = (PACKET / "handwritten.py").read_text()
    assert text.count("@triton.jit") == 2
    quiz = json.loads((ROOT / "quiz.json").read_text())
    assert len(quiz["questions"]) == 6
