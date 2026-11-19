"""The M2 device matrix is validated on CPU; measurements need CUDA."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_device_matrix_validates_without_torch_or_cuda():
    result = subprocess.run(
        [sys.executable, "benchmarks/competitive/device_runner.py", "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "6 cases" in result.stdout


def test_device_cases_have_outputs_and_tolerances():
    import json
    payload = json.loads((ROOT / "benchmarks/competitive/cases.json").read_text())
    for case in payload["cases"]:
        assert case["output"] in {"out", "q", "z"}
        assert case["atol"] >= 0 and case["rtol"] >= 0
