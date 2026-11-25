"""PyTorch statement of Lesson 2."""
import torch


def layout_mix(x):
    a, b = torch.split(x, x.shape[1] // 2, dim=1)
    interleaved = torch.stack((a, b), dim=-1).reshape(x.shape)
    joined = torch.cat((a, b), dim=1)
    return interleaved + joined
