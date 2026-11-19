# Runtime dependency audit (2026-11-19)

A clean Python 3.11 environment was created from the 0.16.3 wheel and audited
with `pip-audit 2.10.1`:

```text
No known vulnerabilities found
flowgpu  Dependency not found on PyPI and could not be audited: flowgpu (0.16.3)
```

The `flowgpu` skip is expected because the package is not published on PyPI yet;
the only runtime dependency, `lark`, was audited. The repository now runs this
audit automatically in the `security-audit` workflow.
