from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_public_boundary_check_passes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "check_public_boundary.py")],
        cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
