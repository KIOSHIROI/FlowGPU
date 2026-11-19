# Maintenance and release process

## Cadence

| Check | When |
|---|---|
| CI (tests + coverage gate, lint, docs, wheel) | every push / pull request, plus weekly schedule |
| `tools/maintenance_check.py` | release and scheduled maintenance |
| Dependabot updates | weekly, grouped where practical |
| Issue/PR triage | at least weekly |
| Security advisories / CodeQL | as reported |
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
   that the tag matches the package version, builds sdist+wheel, keeps the
   artifacts and creates a GitHub Release with generated notes, both files and
   a `SHA256SUMS` checksum file.
5. Do not publish to PyPI automatically. `publish-pypi.yml` is manual and guarded;
   configure a PyPI trusted publisher and the `pypi` environment before using it.

## Public/private boundary

This repository is a curated standalone public tree. Private competition
material, organizer task packages, evidence runs, research data and personal
infrastructure are intentionally absent. Never copy them into this repository;
`tools/publish_guard.py` checks the known private paths mechanically.

## Docs site

`mkdocs build --strict` runs in CI. After enabling GitHub Pages (Settings -> Pages -> GitHub Actions), run the manual `docs-pages` workflow to deploy. The site is built from `docs/user-guide/`.

## Dependency review

`dependency-review-action` requires the repository dependency graph to be enabled in GitHub Settings -> Code security and analysis. It is not enabled in this repository, so the workflow was removed rather than shipping a permanently failing check. Once enabled, re-add the standard `pull_request` dependency-review workflow.

## Dependency audit

Weekly and on dependency-file changes, `security-audit` builds the wheel, installs it into a clean runtime environment and runs `pip-audit` over the runtime site-packages. It is intentionally separate from CodeQL: CodeQL checks code, pip-audit checks known dependency advisories.
