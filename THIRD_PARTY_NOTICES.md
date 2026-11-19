# Third-party notices

FlowGPU itself is MIT licensed. See [LICENSE](LICENSE).

| component | license | use |
|---|---|---|
| lark | MIT | LALR parser frontend |
| PyTorch / torch_npu | BSD-3-Clause | imported only by *generated* host code |
| Triton | MIT | imported only by generated/device code |
| pytest, pytest-cov, ruff, build, wheel, setuptools | MIT / PSF-compatible | development and CI only |

The compiler wheel itself depends only on `lark`. Generated modules import
PyTorch and Triton at runtime; redistributing those generated modules is subject
to their licenses.

This repository does not include organizer task packages, competition reports,
or private benchmark evidence.
