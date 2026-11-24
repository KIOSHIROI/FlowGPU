"""Token selection for a generation loop.

The compiler does not own logits or a model; these helpers turn the host's
logits into token ids. They are intentionally inference-only and work on the
last axis, so a caller can use ``[vocab]`` or ``[batch, vocab]`` logits.
"""
from __future__ import annotations


def greedy(logits):
    """Return ``argmax`` over the last axis."""
    import torch
    return torch.argmax(logits, dim=-1)


def sample(logits, *, temperature=1.0, top_k=None, generator=None):
    """Sample token ids from logits with optional temperature and top-k.

    ``temperature <= 0`` is greedy. ``top_k`` is applied before sampling and
    must be a positive integer no larger than the vocabulary. The result uses
    ``torch.int64`` and keeps every leading axis.
    """
    import torch
    if temperature <= 0:
        return greedy(logits)
    if top_k is not None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if top_k > logits.shape[-1]:
            raise ValueError("top_k cannot exceed the vocabulary size")
    scaled = logits.float() / float(temperature)
    if top_k is not None:
        values, indices = torch.topk(scaled, top_k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1,
                                   generator=generator)
        choice = choice.reshape(probs.shape[:-1] + (1,))
        return torch.gather(indices, -1, choice).squeeze(-1).to(torch.int64)
    probs = torch.softmax(scaled, dim=-1)
    flat = probs.reshape(-1, probs.shape[-1])
    choice = torch.multinomial(flat, 1, generator=generator)
    return choice.reshape(probs.shape[:-1]).to(torch.int64)


__all__ = ["greedy", "sample"]
