# Roadmap

## Current status

- [x] single `flowgpu` namespace and `python -m flowgpu`
- [x] Python 3.11–3.14 CI (Linux + macOS)
- [x] GitHub Pages user guide and release verification docs
- [x] checksums, build provenance, SBOM and release workflow
- [ ] TestPyPI/PyPI first publish (trusted publisher configured externally)
- [ ] Windows CI (documented as WSL2 for now)


FlowGPU is a compiler prototype. Near-term work is grouped by contract rather
than by benchmark score.

1. **Stable public API** — keep `compile_source`/`compile_pipelines` small and
   documented; make diagnostics actionable.
2. **Backend contracts** — add measured `PlatformConfig` contracts instead of
   guessing across chips; keep unsupported combinations fail-closed.
3. **Middle-end quality** — layout/materialization decisions, fusion boundaries,
   reduction/scan composition, buffer reuse and cost models.
4. **Evidence** — reproducible correctness and performance artifacts, with
   generated/hand-written attribution separated.
5. **Packaging and docs** — reproducible wheels, Python 3.11–3.13 CI, a docs
   site, nightly GPU smoke when a runner is available.

The detailed roadmap is
[ROADMAP.md](https://github.com/KIOSHIROI/FlowGPU/blob/main/docs/ROADMAP.md).
