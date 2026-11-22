"""GraphRunner argument validation; capture itself needs CUDA."""
import pytest

pytest.importorskip("torch")

from flowgpu.compiler.runtime.graph import GraphRunner


class _Plan:
    def allocate_buffers(self):
        raise AssertionError("no capture should happen in these tests")

    def run_into(self, *args, buffers):
        raise AssertionError("no capture should happen in these tests")


def test_graph_runner_requires_cuda_tensor_inputs():
    with pytest.raises(ValueError, match="CUDA tensors"):
        GraphRunner(_Plan(), (object(),))


def test_graph_runner_requires_inputs():
    with pytest.raises(ValueError, match="at least one input"):
        GraphRunner(_Plan(), ())
