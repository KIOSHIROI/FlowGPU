#!/usr/bin/env python3
"""Check the public tree for private paths, credentials and organizer data.

This complements publish_guard.py: publish_guard refuses known private
directories, while this check looks for personal infrastructure paths and
credential-shaped content anywhere in the tree.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "site", "dist", "build", "__pycache__", ".pytest_cache", ".ruff_cache"}
PATTERNS = {
    "personal host path": re.compile(r"/data/baixy26|baixy26@|192\.168\.\d+\.\d+|/Users/[A-Za-z0-9._-]+/"),
    "ssh material": re.compile(r"BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY|~/.ssh/"),
    "competition credential": re.compile(r"flagos_token|cannbench_token|team_id\s*[:=]|submission_id\s*[:=]"),
    "internal commercial pricing": re.compile(r"person-days|gross_margin|cost floor|price ladder|price_menu|CNY"),
    "internal product positioning": re.compile(r"customer workspace|live engagement|internal roadmap"),
}
FORBIDDEN_FILES = {"golden.py", "proto.yaml", "cases.yaml"}
FORBIDDEN_DIRS = ("competition", "results", "research", "data/cannbench", "presentation")


def main() -> int:
    problems = []
    for relative in FORBIDDEN_DIRS:
        if (ROOT / relative).exists():
            problems.append(f"private path present: {relative}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name == "check_public_boundary.py" or path.name.startswith(".coverage"):
            continue
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_FILES:
            problems.append(f"organizer-shaped file: {relative}")
        if path.suffix in {".pem", ".key", ".p12", ".pfx"}:
            problems.append(f"key-like file: {relative}")
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                problems.append(f"{label} in {relative}:{line}")
    if problems:
        print("public boundary check FAILED")
        for problem in problems:
            print(" -", problem)
        return 1
    print("public boundary check ok: no personal paths, credentials or private directories")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
