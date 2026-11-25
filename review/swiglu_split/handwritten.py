"""Hand-written two-kernel version: split is materialized, then pointwise."""
import torch
import triton
import triton.language as tl


@triton.jit
def split_copy_kernel(x_ptr, gate_ptr, up_ptr, WIDTH: tl.constexpr, HALF: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, HALF)
    tl.store(gate_ptr + row * HALF + column, tl.load(x_ptr + row * WIDTH + column))
    tl.store(up_ptr + row * HALF + column,
             tl.load(x_ptr + row * WIDTH + HALF + column))


@triton.jit
def swiglu_kernel(gate_ptr, up_ptr, alpha_ptr, beta_ptr, limit_ptr, out_ptr,
                  HALF: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, HALF)
    gate = tl.load(gate_ptr + row * HALF + column)
    up = tl.load(up_ptr + row * HALF + column)
    limit = tl.load(limit_ptr)
    alpha = tl.load(alpha_ptr)
    beta = tl.load(beta_ptr)
    upper = tl.minimum(gate, limit)
    bounded = tl.minimum(tl.maximum(up, -limit), limit)
    out = (upper * tl.sigmoid(alpha * upper)) * (bounded + beta)
    tl.store(out_ptr + row * HALF + column, out)


def run_handwritten(x, alpha, beta, limit):
    rows, width = x.shape
    half = width // 2
    gate = torch.empty(rows, half, device=x.device, dtype=x.dtype)
    up = torch.empty_like(gate)
    out = torch.empty_like(gate)
    split_copy_kernel[(rows,)](x, gate, up, width, half, num_warps=4)
    swiglu_kernel[(rows,)](gate, up, alpha, beta, limit, out, half, num_warps=4)
    return out
