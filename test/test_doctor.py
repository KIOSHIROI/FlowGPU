import subprocess
import sys


def test_doctor_prints_environment_without_a_program():
    result = subprocess.run(
        [sys.executable, "-m", "flowgpu", "--doctor"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FlowGPU" in result.stdout
    assert "Python:" in result.stdout
    assert "Reference backend: available" in result.stdout
    assert "lark:" in result.stdout
