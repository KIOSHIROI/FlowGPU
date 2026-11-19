# Dependency triage (2026-11-19)

Dependabot opened its first weekly PRs after the public repository went live.
All three are GitHub Actions major-version bumps and all PR checks are green.

| PR | Update | Files | State |
|---|---|---|---|
| #1 | `actions/setup-python` 5 → 7 | ci / release / nightly workflows | clean, checks green |
| #2 | `actions/upload-artifact` 4 → 7 | ci / release workflows | clean, checks green |
| #3 | `actions/checkout` 4 → 7 | ci / release / nightly workflows | clean, checks green |

Review notes:

- No source, test or packaging changes; only workflow action versions.
- The workflows use the stable inputs (`python-version`, `path`, `name`).
- Merge after confirming the repository's Actions runner images support the
  new major versions; GitHub-hosted runners currently do.
- Future action updates are grouped by the Dependabot configuration.

No security advisories were reported for the runtime dependency (`lark`) at the
time of review. The `pyproject.toml` dev extras remain pinned to compatible
minimal versions.

## Grouped follow-up

Dependabot regrouped the three action bumps into PR #5
(`github-actions` group: `actions/checkout`, `actions/setup-python`,
`actions/upload-artifact`). Its CI run is green. It remains open because the
maintainer session has SSH push access but no GitHub API token for merging or
closing pull requests.

## Dependency graph note

`dependency-review-action` cannot run because the repository dependency graph is not enabled. The workflow was removed; Dependabot and CodeQL remain active. Re-add dependency review after enabling dependency graph in repository security settings.
