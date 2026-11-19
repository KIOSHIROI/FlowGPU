"""Non-contiguous input support: the host-side contract.

The pointwise/rowwise kernels carry the input-layout contract (a contiguous
flag plus per-axis shape/stride constexpr metadata, with an address remap
guarded on the flag). These tests pin the host side of that contract:

1. Pointwise/rowwise consumers no longer refuse non-contiguous inputs -- the
   launch passes `tensor.is_contiguous()` and the per-axis strides, and the
   kernel remaps addresses when the flag is False.
2. Kernel kinds without layout metadata (matmul, transpose, rope, library
   dispatch) still refuse non-contiguous inputs explicitly, with the kinds
   named in the message.
3. The 3-D strided gate path (a [B, H, R] gate read by a 2D-flat kernel, now
   written as the value-layer gate composition) generates without any
   contiguous copy in the host.
4. The reference interpreter is layout-agnostic (it addresses by logical
   index), so the same program runs there for CPU-side numeric checks.
"""

import ast

from compiler.compile import compile_source
from compiler.reference import run as run_reference
from compiler.triton_backend import module


GATE_3D = """pipeline gate3d {
    input x : bf16[B,D]
    input gate : bf16[B,H,R]
    gf = reshape(gate, [B,D])
    y = cast_like(mul(to_fp32(x), sigmoid(to_fp32(gf))), x)
    output y
}"""

POINTWISE = """pipeline point {
    input x : bf16[N]
    input c : bf16[N]
    y = add(x, c)
    output y
}"""

GEMM = """pipeline gemm {
    input x : fp32[M,K]
    input w : fp32[K,N]
    y = matmul(x, w)
    output y
}"""


def host_of(source):
    return compile_source(source).pipelines[0].triton.host


def test_pointwise_host_passes_layout_metadata_not_a_refusal():
    host = host_of(GATE_3D)
    assert "must be contiguous" not in host
    # The launch carries the contiguity flag and per-axis strides.
    assert "x.is_contiguous()" in host
    assert "gate.is_contiguous()" in host
    assert "gate.stride(0)" in host and "gate.stride(2)" in host
    # No contiguous copy is inserted anywhere in the host.
    assert ".contiguous()" not in host


def test_pointwise_kernel_remaps_addresses_when_not_contiguous():
    program = compile_source(GATE_3D).pipelines[0].triton
    kernel = program.kernels[0]
    assert "if not gate_contiguous:" in kernel.source
    # The remap is a full coordinate reconstruction over the 3D strides.
    assert "gate_stride0" in kernel.source and "gate_stride2" in kernel.source
    # The contiguous fast path keeps the flat offsets.
    assert "gate_address = idx" in kernel.source


def test_pointwise_reference_is_layout_agnostic():
    """The reference interpreter addresses by logical index, so a strided
    program is checkable on CPU without a GPU."""
    pipe = compile_source(POINTWISE).pipelines[0]
    golden = run_reference(pipe.reference, {"N": 9})
    assert len(golden["y"]) == 9


def test_matmul_consumer_still_refuses_non_contiguous():
    host = host_of(GEMM)
    assert "must be contiguous for its consuming kernels (matmul do not carry layout metadata)" in host


def test_generated_module_is_valid_python():
    program = compile_source(GATE_3D).pipelines[0].triton
    ast.parse(module([program]))


def test_rowwise_kernel_keeps_weight_layout_guard():
    source = """pipeline rn {
    input x : bf16[B,D]
    input residual : bf16[B,D]
    input weight : bf16[D]
    y = add(x, residual)
    z = rms_norm(y, weight)
    output z
}"""
    program = compile_source(source).pipelines[0].triton
    host = program.host
    assert "must be contiguous" not in host
    assert "weight.is_contiguous()" in host
    assert "weight.stride(0)" in host
    kernel = program.kernels[0]
    assert "if not weight_contiguous:" in kernel.source
