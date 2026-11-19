from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_public_maintenance_check_passes():
    result = subprocess.run([sys.executable, str(ROOT / "tools" / "maintenance_check.py")],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_version_is_consistent():
    import ast
    import tomllib
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    tree = ast.parse((ROOT / "compiler" / "__init__.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            assert ast.literal_eval(node.value) == declared
