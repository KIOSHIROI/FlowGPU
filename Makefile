PY ?= python

.PHONY: install dev test lint docs build check clean

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev,docs]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

docs:
	$(PY) -m mkdocs build --strict

build:
	$(PY) -m build

check: lint test

clean:
	rm -rf build dist .pytest_cache .ruff_cache site
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
