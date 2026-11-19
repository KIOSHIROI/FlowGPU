from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_demo_runs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples" / "quickstart" / "demo.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pipeline: quickstart_add" in result.stdout
    assert "kernels:  1" in result.stdout
