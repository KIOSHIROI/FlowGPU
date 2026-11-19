# Maintenance and release process

## Cadence

| Check | When |
|---|---|
| CI (tests, lint, wheel) | every push / pull request |
| Dependabot updates | weekly, grouped where practical |
| Issue/PR triage | at least weekly |
| Security advisories | as reported |
| Release | when user-facing changes accumulate |

## Required checks

Before tagging:

```sh
python -m pytest -q
python -m ruff check .
python tools/publish_guard.py
uv build
```

The public suite must be green on Python 3.11, 3.12 and 3.13 in CI. The wheel
must install in a clean environment and `from compiler import compile_source`
must work.

## Release

1. Update `CHANGELOG.md` with the user-visible changes.
2. Bump `version` in `pyproject.toml` and `compiler/__init__.py`.
3. Commit, push, wait for CI.
4. Create an annotated tag `vX.Y.Z` and push it. The `release` workflow verifies
   that the tag matches the package version, builds sdist+wheel and keeps the
   artifacts.
5. Do not publish to PyPI automatically. Package-index publication is a separate,
   explicit decision.

## Public/private boundary

This repository is a curated standalone public tree. Private competition
material, organizer task packages, evidence runs, research data and personal
infrastructure are intentionally absent. Never copy them into this repository;
`tools/publish_guard.py` checks the known private paths mechanically.
