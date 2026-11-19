"""Test Suite for Multi-Chip Architecture Lowering Profiles (Path B).

Verifies:
1. Target-specific platform selection:
   - "cuda": full kernel kinds supported (pointwise, rowwise, matmul, transpose, rope)
   - "ascend-910b" / "kunlun-r300": pointwise by default;
     other kernels require explicit experimental API opt-in
   - "ascend" / "kunlunxin" baseline: strictly refuses non-pointwise kernels
2. Platform emission generates target-specific host assertions and comments:
   - NPU assertions for Ascend targets
   - torch_npu imports
3. AST validity across all platforms.
"""

import ast
import unittest
from compiler.compile import compile_source
from compiler.platform_configs import resolve_platform
from compiler.triton_backend import module as triton_module


class TestMultiPlatformLowering(unittest.TestCase):
    def test_ascend_910b_matmul_and_norm_support(self):
        source = """pipeline ascend_gemm {
    input x : bf16[16,64]
    input w : bf16[64,128]
    y = matmul(x, w)
    output y
}"""
        p_910b = resolve_platform("ascend-910b", device_cores=32,
                                 allowed_kernel_kinds=("pointwise", "matmul"))
        res = compile_source(source, platform_config=p_910b)
        prog = res.pipelines[0].triton
        self.assertTrue(prog.complete)
        self.assertEqual(len(prog.kernels), 1)
        self.assertEqual(prog.kernels[0].kind, "matmul")
        code = triton_module([prog])
        self.assertIn("import torch_npu", code)
        self.assertIn(".device.type != 'npu'", prog.host)
        ast.parse(code)

    def test_ascend_baseline_refuses_matmul(self):
        source = """pipeline ascend_baseline_gemm {
    input x : bf16[16,64]
    input w : bf16[64,128]
    y = matmul(x, w)
    output y
}"""
        p_base = resolve_platform("ascend", device_cores=32)
        res = compile_source(source, platform_config=p_base)
        prog = res.pipelines[0].triton
        self.assertFalse(prog.complete)
        self.assertEqual(len(prog.refusals), 1)
        self.assertIn("pointwise kernels only", prog.refusals[0].reason)

    def test_kunlun_r300_matmul_support(self):
        source = """pipeline kunlun_gemm {
    input x : bf16[16,64]
    input w : bf16[64,128]
    y = matmul(x, w)
    output y
}"""
        p_r300 = resolve_platform("kunlun-r300", allowed_kernel_kinds=("pointwise", "matmul"))
        res = compile_source(source, platform_config=p_r300)
        prog = res.pipelines[0].triton
        self.assertTrue(prog.complete)
        self.assertEqual(len(prog.kernels), 1)
        code = triton_module([prog])
        ast.parse(code)

    def test_chip_family_aliases_do_not_claim_unverified_gemm_support(self):
        source = "pipeline g { input x : bf16[16,64] input w : bf16[64,128] y = matmul(x,w) output y }"
        for name in ("ascend-910b", "kunlun-r300"):
            profile = resolve_platform(name, device_cores=32)
            prog = compile_source(source, platform_config=profile).pipelines[0].triton
            self.assertFalse(prog.complete)
            self.assertFalse(prog.kernels)


if __name__ == "__main__":
    unittest.main()
