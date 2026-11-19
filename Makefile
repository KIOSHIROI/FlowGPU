PY ?= python

.PHONY: install dev test lint docs links build check release-check clean

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

links:
	$(PY) tools/check_docs_links.py

build:
	$(PY) -m build

check: lint test

release-check: lint test docs links build
	python tools/check_release_metadata.py --dist dist

clean:
	rm -rf build dist .pytest_cache .ruff_cache site
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
