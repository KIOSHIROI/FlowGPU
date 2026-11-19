#!/usr/bin/env python3
"""Inspect a built wheel for public-release invariants."""
from __future__ import annotations

import argparse
import pathlib
import sys
import tarfile
import zipfile


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", default="dist")
    args = parser.parse_args(argv)
    wheels = sorted(pathlib.Path(args.dist).glob("*.whl"))
    if len(wheels) != 1:
        print(f"expected exactly one wheel in {args.dist}, found {len(wheels)}")
        return 1
    wheel = wheels[0]
    problems = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        top = [name for name in names if name.endswith("top_level.txt")]
        if len(top) != 1:
            problems.append("missing top_level.txt")
        else:
            packages = archive.read(top[0]).decode().split()
            if packages != ["flowgpu"]:
                problems.append(f"top-level packages are {packages}, expected ['flowgpu']")
        if not any(name.endswith("flowgpu/compiler/grammar.lark") for name in names):
            problems.append("grammar.lark is missing from flowgpu.compiler")
        if "flowgpu/py.typed" not in names:
            problems.append("PEP 561 marker flowgpu/py.typed is missing")
        metadata = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            problems.append("missing or duplicate METADATA")
        entry_points = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
        if len(entry_points) != 1:
            problems.append("missing entry_points.txt")
        else:
            text = archive.read(entry_points[0]).decode()
            for expected in ("flowgpu = flowgpu.compiler.cli:main",
                             "flowgpu-migration = flowgpu.migration.cli:main"):
                if expected not in text:
                    problems.append(f"missing console script: {expected}")
    sdists = sorted(pathlib.Path(args.dist).glob("*.tar.gz"))
    if len(sdists) != 1:
        problems.append(f"expected exactly one sdist, found {len(sdists)}")
    else:
        with tarfile.open(sdists[0]) as archive:
            names = archive.getnames()
            payload = [name for name in names if name.count("/") >= 1]
            if not any(name.endswith("flowgpu/compiler/grammar.lark") for name in payload):
                problems.append("sdist is missing flowgpu/compiler/grammar.lark")
            if not any(name.endswith("flowgpu/py.typed") for name in payload):
                problems.append("sdist is missing flowgpu/py.typed")
            for name in payload:
                inner = name.split("/", 1)[1]
                if inner.startswith("compiler/") or inner.startswith("migration/"):
                    problems.append(f"sdist contains removed top-level path: {inner}")
                    break
    if problems:
        print("release metadata FAILED")
        for problem in problems:
            print(" -", problem)
        return 1
    print(f"release metadata ok: {wheel.name}, top-level flowgpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
