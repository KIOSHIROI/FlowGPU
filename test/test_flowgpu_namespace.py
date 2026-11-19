import subprocess
import sys

import flowgpu
import flowgpu.compiler


def test_public_namespace_exports_the_compiler_api():
    assert flowgpu.__version__
    result = flowgpu.compile_source(
        "pipeline p { input x:fp16[4] input y:fp16[4] out=add(x,y) output out }")
    assert result.pipelines[0].triton.complete
    assert flowgpu.compiler.compile_source is flowgpu.compile_source


def test_python_dash_m_runs_the_cli():
    result = subprocess.run(
        [sys.executable, "-m", "flowgpu", "--version"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FlowGPU" in result.stdout
