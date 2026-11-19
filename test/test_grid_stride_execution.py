"""The flat capped grid-stride loop, executed (review P1-14, 2026-10-26).

The previous tests around this loop asserted strings a test wrote itself or
looked for a substring, so a mutation that changed the loop *stride* survived
496 grid-related cases.  These two tests pin the contract twice over:

* the AST of the emitted loop must iterate
  `range(tl.program_id(0), end, tl.num_programs(0))` -- the stride is the
  number of programs actually launched;
* executing the emitted source on the CPU Triton double must write every
  output element exactly once.  A stride that skips tiles leaves elements
  unwritten and a stride that revisits one raises "duplicate CPU store", so
  both failure directions are caught.
"""
import ast
import math

import pytest

from flowgpu.compiler.backend import triton_backend
from flowgpu.compiler.backend.platform_configs import PLATFORMS
from flowgpu.compiler.compile import compile_source
from test import segmented_support


def _config(cores=2, block=1024):
    from dataclasses import replace

    return replace(PLATFORMS["cuda"], mapping_strategy="grid_stride_loop",
                   device_cores=cores, preferred_grid_multiplier=1,
                   default_block_size=block)


def _compile(numel, cores=2, block=1024):
    source = (f"pipeline p {{ input x : fp32[{numel}]\n"
              "             y = mul(x, x)\n"
              "             output y }")
    pipeline = compile_source(source, platform_config=_config(cores, block)).pipelines[0]
    assert pipeline.triton.complete, [refusal.message()
                                    for refusal in pipeline.triton.refusals]
    return pipeline


def test_the_emitted_loop_stride_is_the_launched_program_count():
    pipeline = _compile(1000)
    kernel = pipeline.triton.kernels[0]
    tree = ast.parse(kernel.source)
    loops = [node for node in ast.walk(tree)
             if isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
             and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == "range"]
    assert len(loops) == 1, kernel.source
    arguments = loops[0].iter.args
    assert len(arguments) == 3, kernel.source
    start, _end, stride = arguments
    assert isinstance(start, ast.Call) and start.func.attr == "program_id", kernel.source
    assert isinstance(stride, ast.Call), kernel.source
    assert stride.func.attr == "num_programs", kernel.source
    assert isinstance(stride.args[0], ast.Constant) and stride.args[0].value == 0


def test_executing_the_capped_loop_writes_every_element_once():
    numel, block, cores = 5000, 1024, 2
    pipeline = _compile(numel, cores, block)
    kernel = pipeline.triton.kernels[0]
    program = pipeline.triton

    tl = segmented_support.TL()

    class _Jit:
        """A decorator that keeps the emitted function callable."""
        jit = staticmethod(lambda function: function)
        cdiv = staticmethod(lambda a, b: -(-a // b))

    namespace = {"tl": tl, "triton": _Jit(),
                 "torch": segmented_support.types.SimpleNamespace()}
    exec(kernel.source, namespace)
    entry = namespace[kernel.name]

    tiles = math.ceil(numel / block)
    grid = min(tiles, cores)
    x = segmented_support.Buffer([float(index) for index in range(numel)],
                                 (numel,), "fp32")
    y = segmented_support.Buffer([None] * numel, (numel,), "fp32")
    values = {"x": x}
    outputs = {"y": y}

    arguments = []
    for parameter in kernel.parameters:
        base = parameter.split(":")[0].strip()
        if base.endswith("_ptr"):
            value = base[: -len("_ptr")]
            arguments.append(outputs.get(value) or values[value])
        elif "contiguous" in base:
            arguments.append(True)
        elif base.endswith("dim0"):
            arguments.append(numel)
        elif base.endswith("stride0"):
            arguments.append(1)
        elif base in {"numel", "N"}:
            arguments.append(numel)
        elif base.isupper():          # BLOCK and friends
            arguments.append(block)
        else:                          # pragma: no cover - new ABI shape
            raise AssertionError(f"the double does not know argument {parameter!r}")

    tl.grid = (grid, 1, 1)
    for program_id in range(grid):
        tl.pid = (program_id, 0, 0)
        entry(*arguments)

    assert len(y.writes) == numel, (len(y.writes), numel)
    for index in range(numel):
        assert y.data[index] == pytest.approx(float(index) * float(index))
