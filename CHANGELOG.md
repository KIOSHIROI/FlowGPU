# Changelog

## 0.58.0

- Add the first end-to-end decoder layer to M2: graph 32.16 us vs eager
  475.55 us / Inductor 197.63 us.

## 0.57.1

- Align the competitive summary with the 21-case matrix and the measured
  FlagOS/FlagGems baseline; record TVM/TensorRT as external blockers.

## 0.57.0

- Add the FlagOS/FlagGems baseline backend and publish a 10-case comparison.
- Document TVM: the PyPI wheel is CPU-only, so no GPU-comparable baseline.

## 0.56.0

- Declaration-only shape detection: literal-only operators (sort/topk/conv)
  now work from symbolic declarations through per-shape rewriting.
- Verified symbolic sort_last and conv2d across two shapes on H100.

## 0.55.1

- Validate all 21 cases: every backend passes, median compile 11.3 ms.
- Small MoE is within noise of eager; large MoE graph clears both competitors.

## 0.55.0

- Add a larger MoE router case ([32,128], 64 experts, k=8); graph 27.98 us
  vs eager 61.58 us, showing the small-case gap is launch-bound.

## 0.54.2

- Refresh the summary for 20 cases and record the MoE router as an open
  eager/Inductor gap; PyPI remains externally blocked.

## 0.54.1

- Validate the full 20-case matrix: all backends correct, median compile
  15.6 ms. Document the MoE router as a remaining eager/Inductor gap.

## 0.54.0

- Add the QK RMSNorm+RoPE case to M2; graph 6.62 us vs eager 184.5 us.

## 0.53.1

- Refresh the one-page competitive summary for the 19-case matrix, tuning,
  dynamic fallback, and the remaining external baseline/PyPI gaps.

## 0.53.0

- Update compile-time and peak-memory reports to all 19 cases, including MoE
  and top-k; MoE shows the same 32 MiB first-use allocation effect.

## 0.52.1

- Lock ShapeRunner symbolic-to-literal fallback with constructor and runtime
  tests.

## 0.52.0

- ShapeRunner falls back from a symbolic module to literal per-shape rewriting
  when an operator has no symbolic lowering; verified on symbolic mean_last.

## 0.51.1

- Validate the full 19-case matrix in one run: every backend passes, median
  compile 15.0 ms; publish the grouped latency snapshot.

## 0.51.0

- Add the topk_last sampling case to M2; 4.70 us vs eager 18.44 us on H100.

## 0.50.0

- Add the multi-operator MoE router case to M2: matmul + tanh soft-cap +
  softmax + topk + gather; 48-53 us vs eager 151.6 us on H100.

## 0.49.0

- Add conv2d_ex(x, weight, bias, stride, padding), one parameterized
  implicit-GEMM convolution for arbitrary static stride/padding.

## 0.48.0

- Cache tune_conv2d choices by source/shape/platform; a warm call compiles only
  the winning geometry and reports cached=True.

## 0.47.1

- Add flowgpu_tuned to the M2 runner and publish tuned conv results: stride-2
  cases improve from 18-22 us to 3.68-3.74 us; small 1x1 is unchanged.

## 0.47.0

- Add tune_conv2d, a measured conv tile selector. On the 3x3 same case it
  finds a 3.91 us geometry where the default candidate is 19.33 us.

## 0.46.1

- Publish the full 17-case latency/correctness snapshot and mark shared-GPU
  outliers separately from prepared/graph values.

## 0.46.0

- Add conv2d_stride2_same (3x3, stride 2, padding 1); prepared/graph are
  3.56-3.78 us vs eager 39.92 us on H100.

## 0.45.2

- Publish peak-allocated memory for all 16 cases; first-use matmul/conv
  allocations are separated from prepared/static buffers.

## 0.45.1

- Publish the full 16-case matrix and compile-time distribution: 11.7-27.5 ms
  after a cold first compile, median 15.2 ms.

## 0.45.0

- Add conv2d_stride2 (stride-2, no padding) on the implicit-GEMM lowering;
  prepared/graph are 3.7 us vs eager 55.55 us on H100.

## 0.44.0

- Add conv2d_same: odd-K same-padded convolution on the implicit-GEMM
  lowering; 3.90 us vs eager 62.04 us on H100.

## 0.43.1

- Add a CPU Docker image, a CI build job, and container documentation.

## 0.43.0

- Support 4-D [O,C,KH,KW] weights and lower small KxK conv2d to an
  implicit-GEMM tl.dot kernel; 3x3 is 4.03 us vs eager 50.25 us on H100.

## 0.42.1

- Cap conv2d 1x1 tile extents (BM/BN 128, BK 64); a [4,16,16,32] x [64,32]
  case is within fp16 tolerance on H100.

## 0.42.0

- Lower conv2d 1x1 to one tiled tl.dot kernel (2.37 us vs 136 us for the
  four-kernel composition); prepared and graph paths now work.

## 0.41.0

- Extend the M2 matrix to 13 cases with batch_norm and 1x1 conv2d.
- batch_norm is fused and fast; conv2d_1x1 exposes a 136 us four-kernel gap.

## 0.40.1

- Keep a portable layer_norm expansion for symbolic or wider-than-4096 rows;
  literal in-register rows still use the native kernel.

## 0.40.0

- Support rank-0 scalar arguments in prepared execution plans and GraphRunner;
  layer_norm prepares and captures, all four FlowGPU paths pass on H100.

## 0.39.0

- Replace the layer_norm composition with one native Triton row kernel;
  exact for eps 1e-5/1e-3 on H100 and 2.42 us vs eager 36.75 us (shared GPU).

## 0.38.0

- Replace the mean_last composition with one native Triton row-mean kernel.

## 0.37.0

- Replace the quadratic topk sort composition with a native Triton `tl.sort`
  row kernel; sort_last is 2.23 us vs eager 13.87 us on H100.

## 0.36.2

- Publish a one-page competitive capability/latency summary with honest limits.

## 0.36.1

- Add the prepared+graph M2 backend and publish the 9-case comparison;
  copy+replay is 8.56–14.65 µs, faster than allocating prepared but slower than
  zero-copy graph.

## 0.36.0

- Add `prepare_graph_pipeline` combining prepared execution plans with graph
  capture; copy+replay measured 11.12/10.96 µs for add/residual on H100.

## 0.35.1

- Add `ShapeRunner`, which selects symbolic vs literal dynamic handling from
  the program declarations; verified on H100 for both paths.

## 0.35.0

- Add `LiteralShapeRunner`: rewrite named literal input declarations per shape
  and compile a static specialization; verified on H100 for 1024/2048/17.

## 0.34.2

- Add peak-allocated-memory reporting to the M2 device runner and publish the
  9-case result; prepared/static matmul avoids a 32 MiB first-use allocation.

## 0.34.1

- Publish 3-seed medians including the zero-copy graph path: 2.99–4.31 µs
  across all 9 cases.

## 0.34.0

- Add the `flowgpu_graph` M2 backend and publish zero-copy CUDA Graph replay
  across all 9 cases: 2.62–4.30 µs, below eager and explicit Triton.

## 0.33.1

- Measure symbolic vs per-shape compile cost and full-call latency on H100 GPU1;
  one module is not automatically cheaper for a small shape set.

## 0.33.0

- Add `SymbolicShapeRunner` and document one-module runtime-extent dispatch;
  verified on H100 for vectoradd, matmul and softmax across shapes.
- Coverage records dynamic shapes as partial; no checklist gaps remain.

## 0.32.0

- Add `sort_last` as a correctness-first composition (neg + topk(k=width) + neg);
  H100 GPU1 matches torch.sort exactly on literal rows.
- Coverage moves to 45/46; only symbolic dynamic shapes remain.

## 0.31.0

- Add a 1x1 conv2d (NHWC, flattened [O,C] weight) composed from reshape,
  transpose, matmul and bias; H100 GPU1 matches torch exactly.
- Coverage moves to 44/46 with 91 registered operators.

## 0.30.2

- Add a multi-seed M2 wrapper and publish 3-seed medians for all 9 cases;
  all seeds pass correctness.

## 0.30.1

- Share the row-count reduction between layer_norm's mean and variance;
  kernels drop 4 -> 3 and H100 GPU1 full-call is 60.9 us.

## 0.30.0

- Add rank-2 inference `batch_norm` with provided mean/var, affine weight/bias
  and scalar eps; H100 GPU1 matches torch exactly.
- Coverage moves to 43/46 with 90 registered operators.

## 0.29.2

- Extend the M2 matrix to nine cases with `neg`, `pow` and `mean_last`; all
  correctness checks pass and prepared `mean_last` is 12.89 µs on H100 GPU1.

## 0.29.1

- Add `GraphRunner(copy_inputs=False)` zero-copy capture for caller-owned static
  buffers; in-place updates need no per-call copy.

## 0.29.0

- Add `layer_norm` as a composed fp32-mean/variance operator with affine
  weight/bias and scalar eps; H100 GPU1 matches torch layer_norm exactly.
- Coverage moves to 42/46, with 89 registered operators.

## 0.28.0

- Add `mean_last` as a composed operator (sum_last / count of ones), with an
  H100 GPU1 correctness check and coverage at 41/46.

## 0.27.0

- Add `pow` as a value-layer binary operator with torch-compatible NaN
  semantics; H100 GPU1 matches torch.pow exactly, including NaN masks.
- Update the coverage audit to 40/46 present.

## 0.26.0

- Add `neg` as a value-layer unary operator with reference and Triton lowerings;
  H100 GPU1 correctness is exact.
- Update the coverage audit to 39/46 present.

## 0.25.2

- Add and publish the operator-coverage audit (38/46 present, 8 gaps).

## 0.25.1

- Batch GraphRunner input copies with `_foreach_copy_`, reducing copy + replay
  from 9.3–18.3 µs to 8.1–11.2 µs on H100 GPU1.

## 0.25.0

- Add `GraphRunner`: capture a prepared execution plan and replay fixed-shape
  static buffers with no guards, allocation or Triton dispatch.
- Measure graph replay on H100 GPU1 and document the fixed-shape deployment
  contract.

## 0.24.1

- Make the prepared execution-plan path first-class: select one named pipeline
  from a multi-pipeline source, document caller-owned buffers, and publish the
  H100 GPU1 comparative result and optimization order.
- Add the prepared FlowGPU backend to the M2 device runner.
- Add the CPU competitive probe, six-case matrix, independent CPU oracles and
  competitive roadmap; wire compile/oracle checks into CI.
- Add the M2 device runner with shared inputs/tolerance and separate CUDA-Graph
  and full-call metrics for FlowGPU vs eager vs TorchInductor vs hand-written
  Triton baselines.

## 0.24.0

- Add `DynamicShapeRunner`: derive symbolic bindings per call and cache static
  per-shape specializations with explicit rebuilds.
- Document the practical dynamic-shape boundary and M3 status.

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
