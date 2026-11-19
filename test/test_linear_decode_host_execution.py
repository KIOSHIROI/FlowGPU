"""linear_decode_forward: the emitted host and kernel are executed, not read.

The 2026-10-26 review found the previous regression only asserted the strings
`cache.clone()` and `DW: tl.constexpr`, while the kernel still wrote the *input*
cache pointer and the host used `HD` before binding it.  This file executes the
emitted host on the CPU Triton double, runs the emitted kernel over the real
grid, and compares both results against the reference lowering.
"""
import itertools
import types

import pytest

from compiler.backend import reference
from compiler.compile import compile_source
from test import segmented_support


LITERAL = (
    "pipeline ld {\n"
    " input q:fp16[2,2,1,2] input k:fp16[2,2,1,2] input v:fp16[2,2,1,2]\n"
    " input cache:fp16[4,2,2,2] input slope:fp32[2] input slots:int32[2]\n"
    " out, cache_out = linear_decode_forward(q, k, v, cache, slope, slots, 8)\n"
    " output out\n"
    " output cache_out\n"
    "}\n")

SYMBOLIC = (
    "pipeline ld {\n"
    " input q:fp16[B,H,1,D] input k:fp16[B,H,1,D] input v:fp16[B,H,1,D]\n"
    " input cache:fp16[S,H,D,D] input slope:fp32[H] input slots:int32[B]\n"
    " out, cache_out = linear_decode_forward(q, k, v, cache, slope, slots, 8)\n"
    " output out\n"
    " output cache_out\n"
    "}\n")


class _Buffer(segmented_support.Buffer):
    def clone(self):
        copy = _Buffer(self.data, self.shape, str(self.dtype), self.strides)
        copy.device = self.device
        copy.is_cuda = self.is_cuda
        return copy


class _Kernel:
    def __init__(self, function):
        self.function = function

    def __getitem__(self, grid):
        grid = grid if isinstance(grid, tuple) else (grid,)

        def launch(*arguments, **options):
            tl = _HARNESS.tl
            tl.grid = grid + (1,) * (3 - len(grid))
            for program in itertools.product(*(range(int(extent)) for extent in grid)):
                tl.pid = program + (0,) * (3 - len(program))
                self.function(*arguments)
        return launch


class _Triton:
    cdiv = staticmethod(lambda a, b: -(-a // b))
    jit = staticmethod(lambda function: _Kernel(function))


class _Harness:
    def __init__(self):
        self.tl = segmented_support.TL()

    def torch(self):
        def empty(shape, *, dtype, device):
            dtype = str(dtype)
            return _Buffer([0.0] * int(__import__("math").prod(shape)),
                           tuple(shape), dtype)

        return types.SimpleNamespace(
            empty=empty, strided="strided", float16="fp16", bfloat16="bf16",
            float32="fp32", int32="int32", int64="int64", bool="bool")


_HARNESS = _Harness()


def _execute_host(program, tensors):
    """Run the emitted host with the CPU double's tensor/kernel stubs."""
    namespace = {"tl": _HARNESS.tl, "triton": _Triton(),
                 "torch": _HARNESS.torch()}
    for kernel in program.kernels:
        exec(kernel.source, namespace)
    exec(program.host, namespace)
    # linear_decode's host returns a tuple and keeps the pipeline name (the
    # generic host form is `run_<pipeline>`).
    entry = (namespace.get(f"run_{program.pipeline}")
             or namespace.get(program.pipeline)
             or next(value for name, value in namespace.items()
                     if name.startswith("run_")))
    return entry(*tensors)


def _double_tensors():
    q = _Buffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 2, 1, 2), "fp16")
    k = _Buffer([1.0, 0.5, 2.0, 1.5, 0.25, 1.0, 0.75, 2.5], (2, 2, 1, 2), "fp16")
    v = _Buffer([2.0, 1.0, 0.5, 1.5, 3.0, 2.0, 1.0, 4.0], (2, 2, 1, 2), "fp16")
    cache = _Buffer([float(index) * 0.5 for index in range(32)], (4, 2, 2, 2), "fp16")
    slope = _Buffer([0.0, 1.0], (2,), "fp32")
    slots = _Buffer([0, 1], (2,), "int32")
    return q, k, v, cache, slope, slots


def _reference(program, tensors, dimensions=None):
    namespace = {}
    exec(reference.module([program.reference], numeric_mode="fp32"), namespace)
    entry = next(value for name, value in namespace.items() if name.startswith("run_"))
    arguments = [list(tensor.data) for tensor in tensors]
    if dimensions:
        arguments += [dimensions[name] for name in program.reference.dimensions]
    return entry(*arguments)


#: fp16 storage rounds at the operator boundary; the reference accumulates in
#: double, so agreement is at the dtype's resolution (~1e-3 relative).
TOLERANCE = dict(rel=2e-3)


def test_the_literal_program_writes_its_cache_result_and_leaves_the_input_alone():
    program = compile_source(LITERAL).pipelines[0]
    kernel = program.triton.kernels[0]
    assert "tl.store(new_ptr" in kernel.source
    assert "new_ptr = cache_out_ptr" in kernel.source
    q, k, v, cache, slope, slots = _double_tensors()
    original_cache = list(cache.data)
    out, cache_out = _execute_host(program.triton, (q, k, v, cache, slope, slots))
    assert cache.data == original_cache, "the input cache tensor was mutated"
    dimensions = {"B": 2, "H": 2, "D": 2, "S": 4, "HD": 4}
    expected = _reference(program, (q, k, v, cache, slope, slots), dimensions)
    assert out.data == pytest.approx(expected["out"], **TOLERANCE)
    assert cache_out.data == pytest.approx(expected["cache_out"], **TOLERANCE)


def test_the_symbolic_program_binds_hd_after_h_and_d():
    program = compile_source(SYMBOLIC).pipelines[0]
    host = program.triton.host.splitlines()
    hd = next(index for index, line in enumerate(host) if "HD = H * D" in line)
    heads = next(index for index, line in enumerate(host) if line.strip().startswith("H = q.shape"))
    head_dim = next(index for index, line in enumerate(host) if line.strip().startswith("D = q.shape"))
    assert heads < hd and head_dim < hd
    # and the symbolic program executes too
    q = _Buffer([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], (2, 2, 1, 2), "fp16")
    k = _Buffer([1.0, 0.5, 2.0, 1.5, 0.25, 1.0, 0.75, 2.5], (2, 2, 1, 2), "fp16")
    v = _Buffer([2.0, 1.0, 0.5, 1.5, 3.0, 2.0, 1.0, 4.0], (2, 2, 1, 2), "fp16")
    cache = _Buffer([float(index) * 0.5 for index in range(32)], (4, 2, 2, 2), "fp16")
    slope = _Buffer([0.0, 1.0], (2,), "fp32")
    slots = _Buffer([0, 1], (2,), "int32")
    out, cache_out = _execute_host(program.triton, (q, k, v, cache, slope, slots))
    dimensions = {"B": 2, "H": 2, "D": 2, "S": 4, "HD": 4}
    expected = _reference(program, (q, k, v, cache, slope, slots), dimensions)
    assert out.data == pytest.approx(expected["out"], **TOLERANCE)
    assert cache_out.data == pytest.approx(expected["cache_out"], **TOLERANCE)


@pytest.mark.parametrize("slot_values", [
    [0, -1],        # padding batch: reference leaves the whole output row zero
    [0, 99],        # illegal slot: device must not index cache out of bounds
    [-2, 3],        # negative but not the -1 padding marker
])
def test_padding_and_illegal_slots_write_zero_rows_and_skip_cache(slot_values):
    program = compile_source(LITERAL).pipelines[0]
    kernel = program.triton.kernels[0]
    assert "slot >= S" in kernel.source          # the guard is in the kernel
    q, k, v, cache, slope, _slots = _double_tensors()
    slots = _Buffer(slot_values, (2,), "int32")
    original_cache = list(cache.data)
    out, cache_out = _execute_host(program.triton, (q, k, v, cache, slope, slots))
    assert cache.data == original_cache
    expected = _reference(program, (q, k, v, cache, slope, slots))
    assert out.data == pytest.approx(expected["out"], **TOLERANCE)
    assert cache_out.data == pytest.approx(expected["cache_out"], **TOLERANCE)
    # every output element is written, including the padded batch's row
    assert len(out.writes) == len(out.data)
