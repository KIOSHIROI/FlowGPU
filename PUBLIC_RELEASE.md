# Public release notes

This directory is a standalone, sanitized public snapshot of FlowGPU.

- Source snapshot: FlowGPU compiler prototype, version `1.34.0`.
- The public tree contains the compiler, migration tooling, public examples,
  models, technical documentation, curated CPU tests, license and CI.
- Excluded intentionally: organizer task packages, competition submissions and
  receipts, private benchmark evidence, internal product/strategy material and
  private infrastructure scripts.
- Public test suite: `python -m pytest -q` (CPU only; no GPU required).
- Package build: `uv build` / `python -m build`.
- License: MIT (`LICENSE`).

The release is prepared from a clean commit in the private development tree.
Do not add private evidence back to this tree; `tools/publish_guard.py` checks
the main private paths mechanically.
