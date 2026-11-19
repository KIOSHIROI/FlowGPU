# Changelog

## 0.16.3

No runtime code changes. This release packages the documentation, security
policy, maintenance tooling and CI quality gates added after 0.16.2.

- Add a security policy with supported versions and disclosure expectations.
- Add a stability/versioning guide and CodeQL scanning.
- Add an automated relative-link check for README and the user guide.
- Enforce the configured coverage gate in CI and report coverage artifacts.

## 0.16.2

- Add maintainer, maintenance and release-process documentation.
- Add Dependabot and issue templates for ongoing maintenance.
- Add weekly scheduled CI, a maintenance invariant check and a PR checklist.
- Add a user guide, FAQ, troubleshooting, Python API and CLI docs, and a
  strict MkDocs documentation site build.
- Improve package metadata for discoverability and add `docs` extras.
- Add contributor/roadmap/architecture pages, issue contact links and a guarded
  manual PyPI publishing workflow.
- Add a one-command quickstart demo, a five-minute notebook and manual GitHub
  Pages deployment workflow.
- Finish the user-facing README and package metadata cleanup.

## 0.16.1

- Fix the release/CI wheel smoke check.
- Keep the public CPU test suite green.

## 0.16.0

First public snapshot of the FlowGPU compiler prototype.

- DSL frontend, typed dataflow IR, fusion, buffer reuse, Triton and reference backends.
- CPU test suite, examples, backend contracts and contribution guide.
- Competition/private evidence is intentionally not part of this release.
