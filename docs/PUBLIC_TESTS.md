# Public test suite

The public suite is a curated subset of the project's internal CPU suite. It
runs without GPU hardware, organizer task packages, or private evidence files.

```sh
python -m pytest -q
python -m ruff check .
```

Tests that require private competition/evidence fixtures are intentionally not
part of this repository. The included suite currently contains 1,666 passing CPU tests and exercises the parser, type system,
IR, fusion, buffer planning, layout operators, frontends, reference semantics,
Triton source emission, scalar placement and public API.
