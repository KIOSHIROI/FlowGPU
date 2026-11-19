"""Symbolic-dispatch bindings and the per-shape runner."""
from pathlib import Path
from unittest.mock import patch

import pytest

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.runtime.dynamic_shape import DynamicShapeRunner, derive_bindings

ROOT = Path(__file__).resolve().parents[1]
MATMUL = (ROOT / "examples" / "gpu_mode" / "matmul.fgpu").read_text()


class _Tensor:
    def __init__(self, shape, dtype="float32", device="cuda:0", ptr=0):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._ptr = ptr

    def stride(self):
        return tuple(1 for _ in self.shape)

    def data_ptr(self):
        return self._ptr


def _graph(source, **options):
    return compile_source(source, emit=False, **options).pipelines[0].dataflow


def test_matmul_bindings_come_from_the_actual_extents():
    graph = _graph(MATMUL)
    bindings = derive_bindings(graph, (_Tensor((64, 128)), _Tensor((128, 32))))
    assert bindings == {"M": 64, "K": 128, "N": 32}


def test_symbolic_mismatch_and_literal_mismatch_are_caller_errors():
    graph = _graph(MATMUL)
    with pytest.raises(ValueError, match="binds to both"):
        derive_bindings(graph, (_Tensor((64, 128)), _Tensor((64, 32))))
    add_graph = _graph((ROOT / "examples" / "quickstart" / "add.fgpu").read_text())
    with pytest.raises(ValueError, match="expects extent 1024"):
        derive_bindings(add_graph, (_Tensor((1024,)), _Tensor((8,))))


def test_runner_rebuilds_on_shape_change_and_reuses_metadata(tmp_path):
    calls = []

    def fake_compile(source, *, pipeline, cache_dir, shape_bindings, **options):
        calls.append(dict(shape_bindings))
        return lambda *args: {"out": args[0].shape}

    runner = DynamicShapeRunner(MATMUL, "matmul", tmp_path)
    with patch("flowgpu.compiler.runtime.dynamic_shape.compile_pipeline", fake_compile):
        first = runner(_Tensor((64, 128)), _Tensor((128, 32)))
        second = runner(_Tensor((64, 128)), _Tensor((128, 32)))
        third = runner(_Tensor((32, 64)), _Tensor((64, 16)))
    assert first == second == {"out": (64, 128)}
    assert third == {"out": (32, 64)}
    assert calls == [
        {"M": 64, "K": 128, "N": 32},
        {"M": 32, "K": 64, "N": 16},
    ]
    assert runner.misses == 2 and runner.hits == 1


def test_runner_rejects_caller_supplied_bindings(tmp_path):
    with pytest.raises(ValueError, match="derives shape_bindings"):
        DynamicShapeRunner(MATMUL, "matmul", tmp_path, options={"shape_bindings": {"M": 1}})
