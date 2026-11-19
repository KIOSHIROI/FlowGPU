"""Test Suite for GEMM Group Swizzling & Mapping Configurations.

Verifies:
1. Baseline direct 2D grid generation (group_m = 1).
2. Grouped 1D swizzled grid generation (group_m = 8).
3. Exact launch grid arithmetic:
   - direct: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
   - swizzled: (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
GPU numerical execution is covered by tools/run_gemm_benchmark.py, not these tests.
"""

import ast
import unittest
from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.platform_configs import resolve_platform
from flowgpu.compiler.triton_backend import module as triton_module


class TestGemmSwizzling(unittest.TestCase):
    def setUp(self):
        self.source = """pipeline matmul_swizzle_test {
    input a : bf16[64,128]
    input b : bf16[128,64]
    out = matmul(a, b)
    output out
}"""

    def test_direct_gemm_grid(self):
        p_direct = resolve_platform("cuda", group_m=1)
        res = compile_source(self.source, platform_config=p_direct)
        code = triton_module([res.pipelines[0].triton])
        # Direct grid should be 2D tuple
        self.assertIn("(triton.cdiv(64, BLOCK_M), triton.cdiv(64, BLOCK_N))", code)
        self.assertNotIn("GROUP_M", code)
        # AST parse check
        ast.parse(code)

    def test_swizzled_gemm_grid(self):
        p_swizzle = resolve_platform("cuda", group_m=8)
        res = compile_source(self.source, platform_config=p_swizzle)
        code = triton_module([res.pipelines[0].triton])
        # Swizzled grid should be 1D product
        self.assertIn("(triton.cdiv(64, BLOCK_M) * triton.cdiv(64, BLOCK_N),)", code)
        self.assertNotIn("GROUP_M = 8", code)
        self.assertEqual(res.pipelines[0].triton.kernels[0].launch[-1].text, "8")
        self.assertIn("num_pid_in_group = GROUP_M * num_pid_n", code)
        self.assertIn("first_pid_m = group_id * GROUP_M", code)
        # AST parse check
        ast.parse(code)

    def test_multiple_programs_keep_distinct_group_and_launch_options(self):
        programs = []
        for group, warps, stages in [(3, 8, 4), (8, 4, 2)]:
            source = self.source.replace("matmul_swizzle_test", f"gemm_group{group}")
            p = compile_source(source, platform_config=resolve_platform(
                group_m=group, num_warps=warps, num_stages=stages)).pipelines[0].triton
            self.assertEqual(p.kernels[0].launch[-1].text, str(group))
            self.assertEqual(dict(p.kernels[0].launch_options),
                             {"num_warps": warps, "num_stages": stages})
            self.assertIn(f"num_stages={stages}", p.host)
            programs.append(p)
        code = triton_module(programs)
        self.assertNotIn("GROUP_M =", code)
        ast.parse(code)


if __name__ == "__main__":
    unittest.main()
