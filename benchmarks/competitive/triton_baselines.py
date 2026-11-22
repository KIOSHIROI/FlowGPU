"""Hand-written Triton baselines for the M2 device matrix.

These are ordinary, deliberately simple kernels (not FlowGPU-generated).  They
exist so the competitive report has an explicit Triton column in addition to
TorchInductor.  The kernels are standard masked pointwise/reduction/matmul
forms; they are not tuned and no performance claim is attached to them.

The module imports Triton lazily so CPU-only CI can validate the matrix without
installing a GPU runtime.
"""
from __future__ import annotations

_KERNELS = None


def _kernels():
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(a, b, out, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(a + offsets, mask=mask)
        y = tl.load(b + offsets, mask=mask)
        tl.store(out + offsets, x + y, mask=mask)

    @triton.jit
    def neg_kernel(a, out, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        tl.store(out + offsets, -tl.load(a + offsets, mask=mask), mask=mask)

    @triton.jit
    def pow_kernel(a, b, out, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(a + offsets, mask=mask).to(tl.float32)
        y = tl.load(b + offsets, mask=mask).to(tl.float32)
        tl.store(out + offsets, tl.extra.cuda.libdevice.pow(x, y), mask=mask)

    @triton.jit
    def mean_kernel(x, out, V: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < V
        values = tl.load(x + row * V + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out + row, tl.sum(values) / V)

    @triton.jit
    def int_pointwise_kernel(x, y, out, n: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        a = tl.load(x + offsets, mask=mask)
        b = tl.load(y + offsets, mask=mask)
        tl.store(out + offsets, (a + b) * b, mask=mask)

    @triton.jit
    def softmax_kernel(x, out, V: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < V
        values = tl.load(x + row * V + offsets, mask=mask, other=-float("inf"))
        peak = tl.max(values)
        exp = tl.exp(values - peak)
        total = tl.sum(exp)
        tl.store(out + row * V + offsets, exp / total, mask=mask)

    @triton.jit
    def matmul_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k in range(0, K, BLOCK_K):
            a_tile = tl.load(a + offs_m[:, None] * K + offs_k[None, :] + k,
                             mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
                             other=0.0)
            b_tile = tl.load(b + (offs_k[:, None] + k) * N + offs_n[None, :],
                             mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
                             other=0.0)
            acc += tl.dot(a_tile, b_tile)
        tl.store(c + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16),
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def residual_norm_kernel(x, residual, weight, out, D: tl.constexpr,
                             BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < D
        xv = tl.load(x + row * D + offsets, mask=mask, other=0.0).to(tl.float32)
        rv = tl.load(residual + row * D + offsets, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
        y = xv + rv
        mean = tl.sum(y * y) / D
        scale = tl.rsqrt(mean + 1e-6)
        tl.store(out + row * D + offsets, (y * scale * wv).to(tl.bfloat16), mask=mask)

    _KERNELS = {
        "add": add_kernel,
        "neg": neg_kernel,
        "pow": pow_kernel,
        "mean": mean_kernel,
        "int_pointwise": int_pointwise_kernel,
        "softmax": softmax_kernel,
        "matmul": matmul_kernel,
        "residual_norm": residual_norm_kernel,
    }
    return _KERNELS


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length() if value > 1 else 1


def run(case_name: str, args):
    import torch
    import triton
    kernels = _kernels()
    if case_name in {"add_literal", "vectoradd_symbolic"}:
        a, b = args
        out = torch.empty_like(a)
        n = a.numel()
        block = 256
        kernels["add"][(triton.cdiv(n, block),)](a, b, out, n, block)
        return {"out": out}
    if case_name == "neg_literal":
        (a,) = args
        out = torch.empty_like(a)
        n = a.numel()
        block = 256
        kernels["neg"][(triton.cdiv(n, block),)](a, out, n, block)
        return {"out": out}
    if case_name == "pow_literal":
        a, b = args
        out = torch.empty_like(a)
        n = a.numel()
        block = 256
        kernels["pow"][(triton.cdiv(n, block),)](a, b, out, n, block)
        return {"out": out}
    if case_name == "mean_last":
        (x,) = args
        rows, width = x.shape
        out = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
        block = _next_power_of_two(width)
        kernels["mean"][(rows,)](x, out, width, block)
        return {"out": out}
    if case_name == "mapping_pointwise":
        x, y = args
        out = torch.empty_like(x)
        n = x.numel()
        block = 256
        kernels["int_pointwise"][(triton.cdiv(n, block),)](x, y, out, n, block)
        return {"z": out}
    if case_name == "softmax_wide":
        (x,) = args
        rows, width = x.shape
        out = torch.empty_like(x)
        block = _next_power_of_two(width)
        kernels["softmax"][(rows,)](x, out, width, block)
        return {"q": out}
    if case_name == "matmul_symbolic":
        a, b = args
        m, k = a.shape
        _, n = b.shape
        out = torch.empty((m, n), dtype=a.dtype, device=a.device)
        grid = (triton.cdiv(m, 16), triton.cdiv(n, 16))
        kernels["matmul"][grid](a, b, out, m, n, k, 16, 16, 32)
        return {"out": out}
    if case_name == "residual_norm":
        x, residual, weight = args
        rows, width = x.shape
        out = torch.empty_like(x)
        block = _next_power_of_two(width)
        kernels["residual_norm"][(rows,)](x, residual, weight, out, width, block)
        return {"z": out}
    raise ValueError("no Triton baseline for " + case_name)


__all__ = ["run"]
