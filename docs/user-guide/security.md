# Security

## Reporting a vulnerability

Do **not** open a public issue for an unfixed vulnerability. Use GitHub's private
vulnerability reporting from the repository's **Security** tab. See
[SECURITY.md](https://github.com/KIOSHIROI/FlowGPU/blob/main/SECURITY.md) for the
supported versions, scope and disclosure expectations.

## What runs automatically

| Check | Trigger |
|---|---|
| CodeQL (Python) | push, pull request, weekly |
| `pip-audit` on a clean runtime install | weekly and dependency changes |
| CycloneDX SBOM upload | with the runtime audit |
| Dependabot | weekly Actions / pip / pre-commit updates |
| GitHub build-provenance attestations | every release |
| `SHA256SUMS` in GitHub Releases | every release |
| CycloneDX SBOM artifact | weekly/security-audit workflow |

Dependency review is not enabled because the repository dependency graph is off;
the audit above covers the published runtime dependency instead.

## Verify a release

Download the wheel from the GitHub Release, then verify the attached checksum:

```sh
sha256sum -c SHA256SUMS
```

If GitHub CLI is available, verify the build provenance:

```sh
gh attestation verify flowgpu-<version>-py3-none-any.whl --repo KIOSHIROI/FlowGPU
```

The release wheel has a single top-level package (`flowgpu`); the project never
ships organizer data, competition evidence or private infrastructure. The
`tools/publish_guard.py` check is part of the release process.

## Running untrusted programs

`--run-gpu` executes generated code on a device. Treat untrusted `.fgpu` input
and generated kernels as untrusted code; the compiler/reference path is the safe
default.

The project has no bug bounty or support SLA.
