from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "qwen2_7b_decode.json"


def test_init_workspace_and_one_page_render(tmp_path):
    workspace = tmp_path / "ws"
    result = subprocess.run(
        [sys.executable, "-m", "flowgpu.migration.cli", "init",
         "--model", str(MODEL), "--platforms", "cuda,ascend", "--out", str(workspace)],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (workspace / "README.md").exists()
    bundle = tmp_path / "bundle"
    result = subprocess.run(
        [sys.executable, "-m", "flowgpu.migration.cli", "bundle", "new",
         "--model", str(MODEL), "--platforms", "cuda", "--out", str(bundle)],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [sys.executable, "-m", "flowgpu.migration.cli", "bundle", "render",
         str(bundle), "--one-page"],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Acceptance decision" in result.stdout
