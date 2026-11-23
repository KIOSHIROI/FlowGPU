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


def _fake_literal(module, calls):
    class FakeLiteral:
        def __init__(self, source, pipeline, cache_dir, options=None):
            calls.append("literal-init")

        def __call__(self, *tensors):
            calls.append("literal-call")
            return {"out": tensors[0]}
    FakeLiteral.__name__ = "LiteralShapeRunner"
    return FakeLiteral


def test_shape_runner_falls_back_when_symbolic_lowering_is_unavailable(monkeypatch):
    from pathlib import Path
    from flowgpu.compiler.runtime import literal_shape, shape_runner, symbolic_shape
    source = (Path(__file__).resolve().parents[1] / "examples" / "gpu_mode" / "vectoradd.fgpu").read_text()
    calls = []

    class BrokenSymbolic:
        def __init__(self, *args, **kwargs):
            raise ValueError("no symbolic lowering")

    monkeypatch.setattr(symbolic_shape, "SymbolicShapeRunner", BrokenSymbolic)
    monkeypatch.setattr(literal_shape, "LiteralShapeRunner", _fake_literal(literal_shape, calls))
    runner = shape_runner.ShapeRunner(source, "vectoradd", ".cache")
    assert runner.mode == "LiteralShapeRunner"
    assert isinstance(runner._fallback_error, ValueError)
    assert runner("x")["out"] == "x"
    assert calls == ["literal-init", "literal-call"]


def test_shape_runner_falls_back_on_a_runtime_symbolic_failure(monkeypatch):
    from pathlib import Path
    from flowgpu.compiler.runtime import literal_shape, shape_runner, symbolic_shape
    source = (Path(__file__).resolve().parents[1] / "examples" / "gpu_mode" / "vectoradd.fgpu").read_text()
    calls = []

    class FailingRuntime:
        def __init__(self, *args, **kwargs):
            calls.append("symbolic-init")

        def __call__(self, *tensors):
            raise RuntimeError("symbolic guard rejected")
    FailingRuntime.__name__ = "SymbolicShapeRunner"

    monkeypatch.setattr(symbolic_shape, "SymbolicShapeRunner", FailingRuntime)
    monkeypatch.setattr(literal_shape, "LiteralShapeRunner", _fake_literal(literal_shape, calls))
    runner = shape_runner.ShapeRunner(source, "vectoradd", ".cache")
    assert runner.mode == "SymbolicShapeRunner"
    assert runner("x")["out"] == "x"
    assert runner.mode == "LiteralShapeRunner"
    assert calls == ["symbolic-init", "literal-init", "literal-call"]
