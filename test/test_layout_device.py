"""Device-side lowering of the four layout operators (opt-in).

The CPU Triton double executes the emitted device copy kernels; the reference
backend (unchanged host lowering) is the oracle.  The default path keeps the
host lowering and therefore the existing emitted bytes.
"""
from __future__ import annotations

import pathlib

from compiler import compile_source
from compiler.backend import reference
from compiler.backend.platform_configs import PLATFORMS, resolve_platform
from compiler.backend.triton_backend import module
from test.segmented_support import Buffer
from test.test_platform_contract_fixtures import _execute_emitted

ROOT = pathlib.Path(__file__).resolve().parents[1]

SOURCE = """
pipeline layouts {
  input a : fp16[2,8]
  input b : fp16[2,8]
  c = concat(a, b)
  x = reshape(c, [2,4,4])
  p, q = split(x)
  y = interleave(p, q)
  output y
}
"""

# One program per operator for execution: the CPU double rejects a buffer that
# is written twice, and 13.7 legitimately reuses slots when lifetimes permit.
SINGLE_OP_SOURCES = {
    "concat": "pipeline p { input a : fp16[2,8] input b : fp16[2,8] "
              "c = concat(a, b) output c }",
    "reshape": "pipeline p { input a : fp16[32] x = reshape(a, [2,4,4]) output x }",
    "split": "pipeline p { input a : fp16[2,4] p0, p1 = split(a) output p0 output p1 }",
    "interleave": "pipeline p { input a : fp16[2,8] input b : fp16[2,8] "
                  "y = interleave(a, b) output y }",
}


def _programs():
    default = compile_source(SOURCE, layout_views=False).pipelines[0]
    device = compile_source(SOURCE, layout_views=False, platform_config=resolve_platform(
        "cuda", layout_on_device=True)).pipelines[0]
    return default, device


def test_the_flag_is_off_by_default():
    assert PLATFORMS["cuda"].layout_on_device is False
    assert resolve_platform("cuda").layout_on_device is False
    _, device_text = _programs()[0:1][0], module([_programs()[0].triton])
    assert "torch.cat" in device_text, "default lowering must stay host-side"


def test_device_lowering_removes_every_host_layout_transform():
    default, device = _programs()
    assert default.triton.complete and device.triton.complete
    text = module([device.triton])
    for banned in ("torch.cat", "torch.stack", "copy_(", "x[...,"):
        assert banned not in text, banned
    layout = [kernel for kernel in device.triton.kernels if kernel.kind == "layout"]
    assert len(layout) == 5, [kernel.ops for kernel in layout]


def test_device_layout_kernels_match_the_reference_backend():
    cases = {
        "concat": ([float(i) * 0.5 for i in range(16)],
                   [-float(i) - 1.0 for i in range(16)], {"a": [2, 8], "b": [2, 8]}),
        "reshape": ([float(i) * 0.25 for i in range(32)], None, {"a": [32]}),
        "split": ([float(i) for i in range(8)], None, {"a": [2, 4]}),
        "interleave": ([float(i) for i in range(16)],
                       [-float(i) - 1.0 for i in range(16)], {"a": [2, 8], "b": [2, 8]}),
    }
    for name, (values, second, shapes) in cases.items():
        source = SINGLE_OP_SOURCES[name]
        default = compile_source(source).pipelines[0]
        device = compile_source(source, platform_config=resolve_platform(
            "cuda", layout_on_device=True)).pipelines[0]
        assert device.triton.complete
        namespace = {}
        exec(reference.module([default.reference]), namespace)
        expected = namespace["run_p"](values, second) if second is not None \
            else namespace["run_p"](values)
        inputs = [Buffer(values, shapes["a"], "fp16")]
        if second is not None:
            inputs.append(Buffer(second, shapes["b"], "fp16"))
        produced, _ = _execute_emitted(module([device.triton]), tuple(inputs), pipeline="p")
        for key, want in expected.items():
            flat = produced[key]
            assert len(flat) == len(want), (name, key)
            for got, target in zip(flat, want):
                assert abs(got - target) < 1e-3, (name, key, got, target)


def test_the_packed_layout_example_has_no_host_packing_with_the_flag():
    source = (ROOT / "examples/packed/fp4_indexer.fgpu").read_text()
    pipeline = compile_source(source, platform_config=resolve_platform(
        "cuda", layout_on_device=True)).pipelines[0]
    assert pipeline.triton.complete, [str(m) for m in pipeline.triton.messages()]
    text = module([pipeline.triton])
    assert "torch.stack" not in text and "torch.cat" not in text
    assert "copy_(" not in text
