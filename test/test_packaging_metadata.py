"""Packaging metadata must be installable by modern setuptools (PEP 639).

The first real pip install on the H100 host failed here: setuptools>=77 rejects
a table-valued `project.license`, while the old table syntax worked only with
older setuptools.  This test pins the installable form.
"""
from __future__ import annotations

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pyproject_uses_pep639_license_and_new_enough_setuptools():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = data["project"]
    assert isinstance(project["license"], str), "project.license must be an SPDX string"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    requires = data["build-system"]["requires"]
    versions = [entry for entry in requires if entry.startswith("setuptools")]
    assert versions and int(versions[0].split(">=")[1]) >= 77, requires
