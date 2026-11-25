"""One SwiGLU source compiles fused or materialized; results agree."""
import json
from pathlib import Path

from flowgpu.compiler.backend import reference
from flowgpu.compiler.compile import compile_source

ROOT = Path("lessons/06-materialized")
BINDINGS = {"B": 2, "D": 8, "H": 16}


def test_same_source_two_kernel_counts():
    source = (ROOT / "lesson.fgpu").read_text()
    fused = compile_source(source, shape_bindings=BINDINGS).pipelines[0]
    material = compile_source(source, shape_bindings=BINDINGS,
                              materialize_all=True).pipelines[0]
    assert len(fused.triton.kernels) == 3
    assert len(material.triton.kernels) == 4
    assert any(kernel.ops == ("silu", "mul") for kernel in material.triton.kernels)


def test_both_modes_match_the_cpu_reference():
    source = (ROOT / "lesson.fgpu").read_text()
    fused = compile_source(source, shape_bindings=BINDINGS).pipelines[0]
    material = compile_source(source, shape_bindings=BINDINGS,
                              materialize_all=True).pipelines[0]
    one = reference.run(fused.reference, sizes=BINDINGS)["out"]
    other = reference.run(material.reference, sizes=BINDINGS)["out"]
    assert one == other
    quiz = json.loads((ROOT / "quiz.json").read_text())
    assert len(quiz["questions"]) == 5
