"""The symbolic probe compiles its matrix on CPU; measurements need CUDA."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_symbolic_probe_matrix_compiles():
    result = subprocess.run(
        [sys.executable, "benchmarks/competitive/symbolic_probe.py", "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "3 programs" in result.stdout
