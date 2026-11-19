# Contributing

FlowGPU is evidence-first: a change is accepted when a failing fixture exists,
the fix makes it pass, and default emitted source stays byte-identical unless the
change is explicitly about emission.

## Setup

```sh
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev,docs]"
.venv/bin/python -m pip install pre-commit
pre-commit install
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
mkdocs build --strict
```

## Rules

1. **Reproduce first.** Every bug fix lands with a test that fails on the parent
   commit.
2. **Refusal over silence.** If the language or a backend cannot express a
   construct, emit a diagnostic rather than a silent approximation.
3. **Byte identity.** Default emitters are pinned by golden tests; changing the
   default requires updating the golden files and the evidence behind the old
   bytes.
4. **Platform claims are measurements.** Add measured contracts with a fixture
   and a document; do not infer one backend's behaviour from another.
5. **Attribution.** Hand-written Triton wrapped by `flowgpu/compiler/extern.py` is
   labelled extern; generated and hand-written results are reported separately.

## Pull requests

Keep changes focused. Include a test and update documentation when behaviour
changes. CI runs the public CPU suite on Python 3.11–3.13, lint, a strict docs
build and a wheel build.

The complete contributor rules are in
[CONTRIBUTING.md](https://github.com/KIOSHIROI/FlowGPU/blob/main/CONTRIBUTING.md).
