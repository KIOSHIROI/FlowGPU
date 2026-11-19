# Changelog

## 0.20.0

- Add `flowgpu --doctor` environment diagnostics and document it.

## 0.19.2

- Add the user-facing security and release-verification page.
- Validate distributions with `twine check` in CI/release/publish workflows.
- Add a VS Code/Codespaces dev container and contributor tooling polish.

## Unreleased

- Validate wheel/sdist metadata with `twine check` in CI, release and
  index-publication workflows.
- Add a VS Code/Codespaces dev container.

## Unreleased

- Add a user-facing security and release-verification page.

## Unreleased

- Add macOS CPU CI coverage and a governance document.

## 0.19.1

- Add the PEP 561 `py.typed` marker and ship it in the wheel.
- Add `.editorconfig` and pre-commit configuration for contributors.

## Unreleased

- Harden workflow permissions and add job timeouts to prevent stuck CI.

## 0.19.0

- Add tested Python 3.14 support to the CI matrix and classifiers.
- Include the Python integration guide/example and release tooling since
  0.18.2.

## Unreleased

- Add a runnable Python-integration example and guide.

## 0.18.2

- Package the 0.18 migration guide, release guide and TestPyPI workflow.
- Point package metadata at the live documentation site.
- Add the wheel release-metadata check to CI.

## Unreleased

- Add upgrade guidance for the 0.18 namespace migration, a release guide,
  TestPyPI trusted publishing, and a wheel metadata invariant check.

## 0.18.1

- Fix the CI coverage source path after the 0.18.0 namespace migration.
- Keep the public suite and the release workflow green on the new layout.

## 0.18.0

- **Breaking:** move the implementation into `flowgpu.compiler` and
  `flowgpu.migration`; the former top-level `compiler` / `migration` packages
  are removed. Import from `flowgpu` and use `python -m flowgpu`.
- The wheel now contains one top-level package, `flowgpu`.

## Unreleased

- Publish the user guide on GitHub Pages and link it from the README.

## Unreleased

- Generate a CycloneDX runtime SBOM in the security-audit workflow.

## 0.17.0

- Add the `flowgpu` public import namespace (`from flowgpu import compile_source`)
  and `python -m flowgpu`, while keeping the legacy `compiler` imports. This is
  the first step of the pre-1.0 namespace transition.

## 0.16.6

- Generate GitHub build-provenance attestations for release wheel/sdist files.
- Document optional attestation verification.

## 0.16.5

- Keep every public example compilable (remove a staged `store_at` sample and
  add a compile-all-examples test).
- Attach `SHA256SUMS` to GitHub Releases.
- Document installing a published release wheel and verifying its checksum.

## 0.16.4

- Add a GitHub Release on tag pushes with generated notes and wheel/sdist assets.
- Add a weekly `pip-audit` runtime-dependency audit.
- Add an sdist install check to CI.
- Update dependency-triage documentation.

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
