"""ShapeRunner selects symbolic vs literal dynamic handling from declarations."""
from pathlib import Path

from flowgpu.compiler.runtime.shape_runner import uses_symbolic_shapes

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_source_is_detected():
    source = (ROOT / "examples" / "gpu_mode" / "vectoradd.fgpu").read_text()
    assert uses_symbolic_shapes(source, "vectoradd") is True


def test_literal_source_is_detected():
    source = (ROOT / "examples" / "quickstart" / "add.fgpu").read_text()
    assert uses_symbolic_shapes(source, "quickstart_add") is False


def test_missing_pipeline_is_an_error():
    source = (ROOT / "examples" / "quickstart" / "add.fgpu").read_text()
    try:
        uses_symbolic_shapes(source, "missing")
    except ValueError as error:
        assert "no pipeline" in str(error)
    else:
        raise AssertionError("missing pipeline must fail")
