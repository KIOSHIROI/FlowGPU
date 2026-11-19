# Governance

FlowGPU is a small, maintainer-led open-source project.

## Roles

- **Maintainers** review and merge changes, triage issues, run releases and own
  the public/private boundary. Current maintainers are listed in
  [MAINTAINERS.md](MAINTAINERS.md).
- **Contributors** submit issues, discussions and pull requests under the MIT
  license. There is no CLA.

## Decisions

Routine changes are decided in pull requests with tests and review. Changes to
the public API, generated-source contracts, release process or the
public/private boundary require explicit maintainer approval and a CHANGELOG
entry. Backend behaviour is accepted only with measured evidence and a fixture;
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Becoming a maintainer

Sustained, high-quality contributions and review participation are the path to
maintainership. Maintainers are added by consensus of the existing maintainers.

## Releases

Only green commits on `main` are tagged. Releases follow
[docs/user-guide/releasing.md](docs/user-guide/releasing.md). PyPI publication is
a separate, explicit trusted-publishing step.

## Security and conduct

Security reports follow [SECURITY.md](SECURITY.md); community expectations are
in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
