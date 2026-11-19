import subprocess
import sys


def test_migration_demo_runs_from_the_public_tree():
    result = subprocess.run(
        [sys.executable, "-m", "flowgpu.migration.cli", "demo"],
        capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "# Coverage:" in result.stdout
    assert "generated" in result.stdout
