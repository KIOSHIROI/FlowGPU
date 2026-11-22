"""The multi-seed M2 wrapper keeps the protocol and reports medians."""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_repeat_is_importable_and_declares_median_schema():
    result = subprocess.run(
        [sys.executable, "-c",
         "import json,pathlib,sys; sys.path.insert(0,'.'); "
         "from benchmarks.competitive.repeat import merge; "
         "source=json.load(open('benchmarks/competitive/measurements/m2-h100-gpu1-20261122.json')); "
         "out=merge([source]); print(out['schema'], len(out['cases']))"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "flowgpu.competitive.multiseed/1" in result.stdout
