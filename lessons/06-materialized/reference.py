"""PyTorch statement of Lesson 5."""
import torch
import torch.nn.functional as F


def fused_mlp(x, w_gate, w_up, w_down):
    gate = x @ w_gate
    up = x @ w_up
    return (F.silu(gate) * up) @ w_down
