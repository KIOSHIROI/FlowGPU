"""Device smoke test for the nightly workflow.

Compiles two small FlowGPU programs, executes the emitted Triton on a real GPU
and compares against torch.  This is the only automated place where generated
device code runs; the CPU suite executes the emitted source through the test
double instead.

Exit codes: 0 success or "no device and --require-device was not given";
2 when --require-device is set but CUDA is unavailable; 1 on a mismatch/error.
"""
from __future__ import annotations

import argparse
import sys

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _exec_module(code, label, namespace):
    """Execute a generated module from a real file (Triton>=3.7 needs source)."""
    import os
    import tempfile
    fd, path = tempfile.mkstemp(prefix=label.strip("<>").replace("-", "_") + "_",
                                suffix=".py")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(code)
        exec(compile(code, path, "exec"), namespace)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
sys.path.insert(0, str(ROOT))

ADD = """
pipeline smoke_add {
 input a:fp16[4096]
 input b:fp16[4096]
 out = add(a, b)
 output out
}
"""

MATMUL = ROOT / "examples/gpu_mode/matmul.fgpu"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-device", action="store_true",
                        help="fail (exit 2) when no CUDA device is present")
    args = parser.parse_args(argv)
    try:
        import torch
    except ImportError:
        print("gpu-smoke: torch is not installed")
        return 2 if args.require_device else 0
    if not torch.cuda.is_available():
        print("gpu-smoke: no CUDA device on this runner; register a self-hosted "
              "GPU runner and set the GPU_RUNNER repository variable to use this workflow")
        return 2 if args.require_device else 0

    from compiler import compile_source
    from compiler.backend.triton_backend import module

    failures = []

    # 1) pointwise add: whole-pipeline generated launcher.
    program = compile_source(ADD).pipelines[0].triton
    assert program.complete, program.messages()
    namespace = {}
    _exec_module(module([program]), "<smoke-add>", namespace)
    a = torch.randn(4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, device="cuda", dtype=torch.float16)
    actual = namespace["run_smoke_add"](a, b)["out"]
    expected = a + b
    if not torch.allclose(actual, expected, atol=1e-3, rtol=1e-3):
        failures.append("add mismatch: max abs diff %.4g"
                        % (actual - expected).abs().max().item())

    # 2) matmul: the module the GPU MODE matmul_v2 lane uses.
    with open(MATMUL, encoding="utf-8") as handle:
        matmul_program = compile_source(handle.read()).pipelines[0].triton
    assert matmul_program.complete, matmul_program.messages()
    ns = {}
    _exec_module(module([matmul_program]), "<smoke-matmul>", ns)
    m, k, n = 256, 256, 256
    x = torch.randn(m, k, device="cuda", dtype=torch.float16)
    w = torch.randn(k, n, device="cuda", dtype=torch.float16)
    actual = ns["run_matmul"](x, w)["out"]
    expected = x @ w
    if not torch.allclose(actual, expected, atol=0.5, rtol=2e-2):
        failures.append("matmul mismatch: max abs diff %.4g"
                        % (actual - expected).abs().max().item())

    # 3) MXFP8 packed layout: the first extern kernel the language states
    # itself.  Scale bytes and data bytes must match the CPU oracle exactly.
    packed = ROOT / "examples/packed/mxfp8_quantize.fgpu"
    if packed.exists():
        packed_program = compile_source(packed.read_text()).pipelines[0].triton
        assert packed_program.complete, packed_program.messages()
        packed_ns = {}
        _exec_module(module([packed_program]), "<smoke-mxfp8>", packed_ns)
        constants = (1e-30, 448.0, -448.0, 0, 1, 23, 255, 0x7FFFFF, 254)
        values = (torch.randn(4, 2, 32, device="cuda", dtype=torch.float32) * 3.0)
        produced = packed_ns["run_mxfp8_quantize"](values, *constants)
        from compiler.packed_layout import quantize_mxfp8
        codes, scales = quantize_mxfp8(values.reshape(-1).cpu().tolist(), [4, 2, 32])
        got_codes = produced["q"].view(torch.uint8).reshape(-1).cpu().tolist()
        got_scales = produced["s"].reshape(-1).cpu().tolist()
        if got_codes != codes:
            failures.append("mxfp8 data bytes differ from the oracle")
        if got_scales != scales:
            failures.append("mxfp8 scale bytes differ from the oracle")

    # 4) NVFP4 quantiser: the promoted task-172 DSL program, byte-exact.
    fp4_path = ROOT / "examples/packed/fp4_indexer.fgpu"
    if fp4_path.exists():
        fp4_program = compile_source(fp4_path.read_text()).pipelines[0].triton
        assert fp4_program.complete, fp4_program.messages()
        fp4_ns = {}
        _exec_module(module([fp4_program]), "<smoke-nvfp4>", fp4_ns)
        fp4_constants = (6.0, 6e-4, -1.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
                         0, 1, -1, 4, 8, 16, 23, 24, 31, 254, 255, 0x7FFFFF)
        fp4_x = torch.randn(4, 4, 32, device="cuda", dtype=torch.float32) * 3.0
        fp4_out = fp4_ns["run_quantize_fp4_indexer"](fp4_x, *fp4_constants)
        from compiler.packed_layout import quantize_nvfp4_indexer
        fp4_codes, fp4_words = quantize_nvfp4_indexer(
            fp4_x.reshape(-1).cpu().tolist(), [4, 128])
        if fp4_out["q"].reshape(-1).cpu().tolist() != fp4_codes:
            failures.append("nvfp4 data bytes differ from the oracle")
        got_fp4_words = [int(word) & 0xFFFFFFFF
                         for word in fp4_out["sf_word"].reshape(-1).cpu().tolist()]
        if got_fp4_words != [word & 0xFFFFFFFF for word in fp4_words]:
            failures.append("nvfp4 scale word differs from the oracle")

    if failures:
        for failure in failures:
            print("gpu-smoke: FAIL", failure)
        return 1
    print("gpu-smoke: OK (add + matmul + mxfp8 + nvfp4 executed on",
          torch.cuda.get_device_name(0) + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
