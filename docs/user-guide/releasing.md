# Releasing

Releases are tag-driven and guarded. PyPI publication is a separate manual
workflow with trusted publishing; a tag never uploads to an index.

## Before a release

```sh
python -m pytest -q
python -m ruff check .
python tools/check_docs_links.py
python tools/maintenance_check.py
python -m mkdocs build --strict
uv build
uvx twine check dist/*
```

Update `CHANGELOG.md`, bump both `pyproject.toml` and
`flowgpu/compiler/__init__.py`, then build and inspect the distribution:

```sh
uv build
python tools/check_release_metadata.py --dist dist
```

The metadata check inspects both the wheel and the sdist: a single `flowgpu`
top-level package, the grammar file, `py.typed`, both console scripts, no removed
top-level `compiler`/`migration` paths and a matching version.

## Tag

```sh
git tag -a vX.Y.Z -m "FlowGPU X.Y.Z public release"
git push origin main
git push origin vX.Y.Z
```

The `release` workflow verifies the tag, builds sdist+wheel, generates
`SHA256SUMS`, attests build provenance and creates a GitHub Release.

## PyPI and TestPyPI

Both index workflows are manual and use OIDC trusted publishing; no API token
is stored in the repository.

### TestPyPI

1. Create a TestPyPI trusted publisher for project `flowgpu` with:
   - owner `KIOSHIROI`, repository `FlowGPU`
   - workflow `publish-testpypi.yml`
   - environment `testpypi`
2. Run the **publish-testpypi** workflow with the existing tag and type
   `publish` in the confirmation field.
3. Install from TestPyPI in a clean environment and run the quickstart.

### PyPI

1. Create the PyPI trusted publisher with:
   - owner `KIOSHIROI`, repository `FlowGPU`
   - workflow `publish-pypi.yml`
   - environment `pypi`
2. Run **publish-pypi** with the tag and the confirmation input.
3. Never run it from a tag that is not green in CI.

The package must be verified on TestPyPI first; the public PyPI upload is the
last step, not the first.
