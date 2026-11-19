# Security policy

## Supported versions

Security fixes are provided for the latest released minor line (currently
**0.16.x**). Older snapshots are not maintained.

## Reporting a vulnerability

Please **do not open a public issue** for an unfixed vulnerability.

Use GitHub's private vulnerability reporting: open the repository's **Security**
tab and choose **Report a vulnerability** (GitHub Security Advisories). Include:

- affected version/commit;
- a minimal reproducer or `.fgpu` program;
- the expected and observed behaviour;
- whether the issue affects the compiler, generated host code, the reference
  backend or the optional device path.

If private reporting is unavailable, open a minimal public issue asking the
maintainer for a private channel; do not include exploit details in that issue.

## Scope

In scope:

- the compiler and public Python API;
- generated host wrappers and Triton source;
- the reference backend and CLI;
- repository automation and release artifacts.

Out of scope for this project's security process:

- vulnerabilities in PyTorch, Triton, lark or the operating system (report them
  upstream);
- running untrusted models/programs on a device: `--run-gpu` executes generated
  code, so treat untrusted input as untrusted code;
- resource-exhaustion reports without a concrete correctness or code-execution
  impact.

## Disclosure and response

This is a volunteer-maintained prototype. We aim to acknowledge a report within
7 days and to agree on a disclosure timeline; fixes are best-effort and there is
no bug bounty or support SLA. Please give us a reasonable window before public
disclosure.

## Automated checks

- CodeQL runs on pushes, pull requests and weekly.
- `security-audit` runs `pip-audit` over a clean install of the released wheel weekly and on dependency changes, and uploads a CycloneDX SBOM artifact.
- Dependabot keeps Actions and Python dependencies current.
