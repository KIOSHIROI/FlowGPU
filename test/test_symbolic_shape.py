"""Symbolic lowering: one compiled module for several runtime extents."""
from pathlib import Path

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.runtime.symbolic_shape import SymbolicShapeRunner

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_source_compiles_without_shape_bindings():
    source = (ROOT / "examples" / "gpu_mode" / "vectoradd.fgpu").read_text()
    compiled = compile_source(source)
    pipeline = compiled.pipelines[0]
    assert pipeline.pipeline.name == "vectoradd"
    assert pipeline.dataflow is not None


def test_symbolic_runner_rejects_shape_bindings(tmp_path):
    source = (ROOT / "examples" / "gpu_mode" / "vectoradd.fgpu").read_text()
    try:
        SymbolicShapeRunner(source, "vectoradd", str(tmp_path),
                            options={"shape_bindings": {"M": 1, "N": 1}})
    except ValueError as error:
        assert "without shape_bindings" in str(error)
    else:
        raise AssertionError("shape_bindings must be rejected")
