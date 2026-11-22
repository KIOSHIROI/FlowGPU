"""The coverage inventory must stay reproducible and explicit about gaps."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_coverage_report_shape():
    result = subprocess.run(
        [sys.executable, "tools/operator_coverage.py", "--json"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert data["schema"] == "flowgpu.operator_coverage/1"
    assert data["summary"]["present"] + data["summary"]["missing"] == data["summary"]["checklist"]
    entries = {entry["flowgpu"]: entry for entry in data["entries"]}
    assert entries["matmul"]["status"] == "present"
    assert entries["neg"]["status"] == "present"
    assert entries["pow"]["status"] == "present"
    assert entries["mean_last"]["status"] == "present"
    assert entries["layer_norm"]["status"] == "present"
    assert entries["batch_norm"]["status"] == "present"
    assert entries["conv2d"]["status"] == "missing"
