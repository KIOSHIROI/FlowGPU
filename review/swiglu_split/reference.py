"""The PyTorch statement of the operator; independent of FlowGPU."""
import torch


def swiglu_oai_split(x, alpha, beta, limit):
    gate, up = torch.split(x, x.shape[-1] // 2, dim=-1)
    upper = torch.minimum(gate, limit)
    lower = torch.maximum(up, -limit)
    bounded = torch.minimum(lower, limit)
    gated = upper * torch.sigmoid(alpha * upper)
    return gated * (bounded + beta)
