# Changelog

## Unreleased

- Add the prepared FlowGPU backend to the M2 device runner and publish the first
  H100 GPU1 comparative result and optimization order.

## 0.24.0

- Add `DynamicShapeRunner`: derive symbolic bindings per call and cache static
  per-shape specializations with explicit rebuilds.
- Document the practical dynamic-shape boundary and M3 status.

## Unreleased

- Add the CPU competitive probe, six-case matrix, independent CPU oracles and
  competitive roadmap; wire compile/oracle checks into CI.
- Add the M2 device runner with shared inputs/tolerance and separate CUDA-Graph
  and full-call metrics for FlowGPU vs eager vs TorchInductor vs hand-written
  Triton baselines.

## 0.23.0

- Add a declarative, contract-filtered tuning-space generator for measured
  cross-domain selection.
- Add `ShapeSpecializedRunner` for bounded per-shape prepared-runner caches.
- Document frozen autotune, per-shape specialization and prepared deployment.

## 0.22.0

- Add an explicit effect/alias model and graph verifier; stateful cache
  operators are never removed or commoned.
- Add a verified pass framework with graph-level CSE, DCE and guarded identity
  folding, plus `--no-optimize` / `--no-identity-folding` controls.
- Add a structural planning summary and `--emit planning`.
- Document the static-only dynamic-shape boundary and the optimization pipeline.

## Unreleased

- Prohibit private evidence-path tokens across the public tree and neutralize the
  public audit helper default output path.

## Unreleased

- Replace private evidence-path references in public technical notes with
  neutral placeholders.

## 0.21.7

- Add `flowgpu --doctor-format json` for machine-readable environment reports.

## 0.21.6

- Validate the release pipeline against the upgraded GitHub Actions and record
  the Dependabot resolution.

## Unreleased

- Apply the grouped Dependabot GitHub Actions upgrades to the workflow files;
  close the stale PR and leave no open pull requests.

## 0.21.5

- Remove internal evidence-path references from public code comments.
- Ship an explicit empty public extern ledger instead of reading a private path.

## 0.21.4

- Fix the release workflow by keeping the runtime SBOM in the weekly
  security-audit artifact instead of the release job.

## 0.21.3

- Attach the runtime CycloneDX SBOM to GitHub Release assets and checksums.

## 0.21.2

- Neutralize the last stakeholder wording in the public migration bundle output;
  extend the boundary check to catch it.

## Unreleased

- Add an explicit limitations page covering platform, API and evidence scope.

## Unreleased

- Extend the relative-link check to all public documentation.

## Unreleased

- Add an examples catalog and record the current roadmap status.

## Unreleased

- Document the tested platform matrix (Linux/macOS; Windows via WSL2).

## Unreleased


## 0.21.1

- Neutralize remaining customer/engagement positioning in the public migration
  tooling and bundled model provenance.
- Add internal-positioning phrases to the public-boundary check.

## 0.21.0

- Remove the internal commercial pricing module and `flowgpu-migration price`
  from the public package.
- Fix `flowgpu-migration bundle new` to emit explicit gaps for every
  operator/platform and update its reproduction command.
- Prohibit internal pricing text in the public-boundary check.

## 0.20.1

- Fix `flowgpu-migration demo` for the public tree: resolve repository paths
  correctly and bundle the demo model in the package.
- Add a public test for the migration demo.

## Unreleased

- Add a public-boundary check for personal paths, credentials and organizer files.

## Unreleased

- Remove stale version numbers from the security policy and getting-started docs.

## Unreleased

- Require release and index-publication tags to be ancestors of `main`.

## Unreleased

- Validate sdist contents alongside the wheel in the release metadata check.

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
