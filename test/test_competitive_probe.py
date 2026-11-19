"""The competitive structural probe stays runnable in CI."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_every_competitive_case_compiles():
    result = subprocess.run(
        [sys.executable, "benchmarks/competitive/probe.py", "--check"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_probe_json_has_a_plan_for_every_case():
    result = subprocess.run(
        [sys.executable, "benchmarks/competitive/probe.py", "--json"],
        cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "flowgpu.competitive.probe/1"
    names = {entry["name"] for entry in payload["results"]}
    assert len(names) >= 6
    for entry in payload["results"]:
        assert "error" not in entry, entry
        assert entry["plan"]["kernels"] >= 1
        assert entry["oracle"]["status"] == "pass", entry["oracle"]
