from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_python_integration_example_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "integration" / "compile_pipeline.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pipeline: example_add" in result.stdout
    assert "kernels: 1" in result.stdout
    assert "reference kernels: 1" in result.stdout
