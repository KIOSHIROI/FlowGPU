"""Greedy/top-k token selection is inference-only and shape-preserving."""
import pytest


def test_greedy_and_topk_shapes():
    torch = pytest.importorskip("torch")
    from flowgpu.compiler.runtime.sampling import greedy, sample

    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]])
    assert greedy(logits).tolist() == [1, 0]
    tokens = sample(logits, temperature=1.0, top_k=2,
                    generator=torch.Generator().manual_seed(7))
    assert tokens.shape == (2,)
    assert tokens.dtype == torch.int64


def test_temperature_zero_is_greedy_and_bad_k_is_rejected():
    torch = pytest.importorskip("torch")
    from flowgpu.compiler.runtime.sampling import greedy, sample

    logits = torch.tensor([[1.0, 3.0, 2.0]])
    assert sample(logits, temperature=0.0).tolist() == greedy(logits).tolist()
    with pytest.raises(ValueError, match="top_k"):
        sample(logits, top_k=0)
    with pytest.raises(ValueError, match="vocabulary"):
        sample(logits, top_k=4)
