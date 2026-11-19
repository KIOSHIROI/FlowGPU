#!/usr/bin/env python3
"""Check relative Markdown links in the user-facing docs.

External links are ignored (the network is not a release dependency); relative
links must resolve inside the checkout. The MkDocs strict build catches links
inside the site; this checker also covers README, CONTRIBUTING, SUPPORT and
SECURITY style pages, which are rendered by GitHub rather than MkDocs.
"""
from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SUPPORT.md",
    ROOT / "SECURITY.md",
    ROOT / "GOVERNANCE.md",
    ROOT / "MAINTAINERS.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "docs" / "user-guide").glob("*.md")),
]
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def main() -> int:
    missing = []
    checked = 0
    for path in FILES:
        if not path.exists():
            missing.append(f"missing file: {path.relative_to(ROOT)}")
            continue
        text = path.read_text(errors="ignore")
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            checked += 1
            resolved = (path.parent / target).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError:
                missing.append(f"{path.relative_to(ROOT)}: {target} escapes the repository")
                continue
            if not resolved.exists():
                missing.append(f"{path.relative_to(ROOT)}: {target}")
    if missing:
        print("docs link check FAILED")
        for item in missing:
            print(" -", item)
        return 1
    print(f"docs link check ok: {checked} relative links in {len(FILES)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
