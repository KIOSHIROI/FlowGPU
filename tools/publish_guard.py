"""Refuse to build a public release while private data is present.

This repository intentionally excludes organizer data, competition reports and
private experiment evidence from public distributions.  None of those paths may
appear in a public sdist/wheel or public repository.  This guard is the mechanical
half of that release decision.

Usage:
    python tools/publish_guard.py                  # report and exit 1 if blocked
    python tools/publish_guard.py --allow-private-data   # print only
    python tools/publish_guard.py --root /some/tree
"""
from __future__ import annotations

import argparse
import pathlib
import sys

#: Paths that must never be included in a public release.  Deliberately broad:
#: clearing a path requires a review, not editing this list.
PRIVATE_PATHS = (
    "data/cannbench",
    "competition",
    "results",
    "presentation",
    "docs/BATCH8_REPAIR_20261004.md",
    "docs/BATCH8_REPAIR_20261004.md",
)
#: Patterns that indicate copied organizer content outside the paths above.
PRIVATE_PATTERNS = (
    "golden.py",
    "proto.yaml",
    "cases.yaml",
    "official-tasks-",
)


def violations(root: pathlib.Path) -> list:
    problems = []
    for relative in PRIVATE_PATHS:
        path = root / relative
        if path.exists():
            problems.append(f"private path present: {relative}")
    for pattern in PRIVATE_PATTERNS:
        for match in root.rglob(pattern):
            text = str(match.relative_to(root))
            if not any(text.startswith(prefix) for prefix in ("data/cannbench", "results", "competition")):
                problems.append(f"organizer-pattern file outside a known private path: {text}")
    return sorted(set(problems))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--allow-private-data", action="store_true",
                        help="report the findings but do not fail (private builds)")
    args = parser.parse_args(argv)
    root = pathlib.Path(args.root)
    problems = violations(root)
    if not problems:
        print("publish-guard: no organizer/private data found; public release is allowed")
        return 0
    print("publish-guard: this tree is NOT publishable as-is:")
    for problem in problems:
        print("  -", problem)
    print("Build a public tree from an allowlist (see docs/COMPLIANCE_REVIEW_20261027.md) "
          "or pass --allow-private-data for a private build.")
    return 0 if args.allow_private_data else 1


if __name__ == "__main__":
    raise SystemExit(main())
