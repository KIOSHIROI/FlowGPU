"""PyTorch statement of Lesson 3."""
import torch


def row_scale(x):
    return x * x.sum(dim=-1, keepdim=True)
