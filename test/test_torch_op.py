"""The compiled subgraph can be exposed as a PyTorch custom op."""
import pytest


def test_register_torch_op_runs_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from flowgpu.compiler.runtime.torch_op import register_torch_op

    def run(left, right):
        return left + right

    op = register_torch_op(
        "add_demo", run, arity=2, output_shape=(2, 2),
        output_dtype=torch.float32, device_type="CPU")
    result = op(torch.ones(2, 2), torch.ones(2, 2))
    assert torch.equal(result, torch.full((2, 2), 2.0))
