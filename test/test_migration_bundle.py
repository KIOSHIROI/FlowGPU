from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "qwen2_7b_decode.json"


def _run(*args):
    return subprocess.run([sys.executable, "-m", "flowgpu.migration.cli", *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_bundle_roundtrip_is_valid_on_the_public_tree(tmp_path):
    bundle = tmp_path / "bundle"
    result = _run("bundle", "new", "--model", str(MODEL),
                  "--platforms", "cuda,ascend", "--out", str(bundle))
    assert result.returncode == 0, result.stdout + result.stderr
    result = _run("bundle", "validate", str(bundle))
    assert result.returncode == 0, result.stdout + result.stderr
    result = _run("bundle", "render", str(bundle))
    assert result.returncode == 0, result.stdout + result.stderr
