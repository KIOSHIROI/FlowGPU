#!/usr/bin/env python3
"""Small, dependency-free maintenance check for the public FlowGPU tree.

Run before a release and from the scheduled CI job.  It does not replace the
test suite; it checks the invariants that are easy to forget: version
consistency, release metadata, private-data boundary and required workflows.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    problems = []
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    declared = pyproject["project"]["version"]
    init_text = (ROOT / "compiler" / "__init__.py").read_text()
    version = None
    for node in ast.walk(ast.parse(init_text)):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
            version = ast.literal_eval(node.value)
    if version != declared:
        problems.append(f"compiler.__version__ {version!r} != pyproject {declared!r}")
    changelog = (ROOT / "CHANGELOG.md").read_text()
    if f"## {declared}" not in changelog and "## Unreleased" not in changelog:
        problems.append("CHANGELOG has neither the current version nor an Unreleased entry")
    for relative in ("LICENSE", "README.md", "CONTRIBUTING.md", "SECURITY.md",
                     "MAINTAINERS.md", "docs/MAINTENANCE.md",
                     ".github/workflows/ci.yml", ".github/workflows/release.yml",
                     ".github/dependabot.yml"):
        if not (ROOT / relative).exists():
            problems.append(f"missing {relative}")
    for private in ("competition", "results", "research", "data/cannbench", "presentation"):
        if (ROOT / private).exists():
            problems.append(f"private path present: {private}")
    guard = subprocess.run([sys.executable, str(ROOT / "tools" / "publish_guard.py")],
                           cwd=ROOT, capture_output=True, text=True)
    if guard.returncode != 0:
        problems.append("publish_guard refused the tree: " + guard.stdout.strip())
    if problems:
        print("maintenance check FAILED")
        for problem in problems:
            print(" -", problem)
        return 1
    print(f"maintenance check ok: version {declared}, workflows and boundary present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
