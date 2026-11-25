"""Map generated kernels back to the DSL statements that produced them."""
from __future__ import annotations

import json

from flowgpu.compiler.compile import compile_source
from flowgpu.compiler.frontend.parser import parse_ast


def _assignment_lines(ast):
    lines = {}
    for statement in ast.statements:
        if type(statement).__name__ != "Assignment":
            continue
        line = getattr(statement.position, "line", None)
        for name in statement.names:
            lines[name] = line
    return lines


def build_source_map(source, *, shape_bindings=None):
    """Return kernel -> source lines for the first pipeline."""
    ast = parse_ast(source)[0]
    value_lines = _assignment_lines(ast)
    result = compile_source(source, shape_bindings=shape_bindings)
    program = result.pipelines[0]
    groups = program.fusion.groups
    kernels = program.triton.kernels
    mapped = []
    for index, kernel in enumerate(kernels):
        group = groups[index] if index < len(groups) else None
        lines = set()
        values = {}
        if group is not None:
            for node in group.members:
                if node.output in value_lines:
                    line = value_lines[node.output]
                    lines.add(line)
                    values[node.output] = line
        mapped.append({
            "kernel": kernel.name,
            "ops": list(kernel.ops),
            "source_lines": sorted(line for line in lines if line is not None),
            "values": values,
        })
    return {"pipeline": program.pipeline.name, "kernels": mapped}


def render_markdown(report, source):
    source_lines = source.splitlines()
    lines = [f"# Source map: {report['pipeline']}", ""]
    for kernel in report["kernels"]:
        lines.append(f"## `{kernel['kernel']}`")
        lines.append(f"- ops: {', '.join(kernel['ops'])}")
        lines.append(f"- DSL lines: {kernel['source_lines']}")
        for number in kernel["source_lines"]:
            if 1 <= number <= len(source_lines):
                lines.append(f"  - L{number}: `{source_lines[number - 1].strip()}`")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse
    import pathlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--shape", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    bindings = {}
    for item in args.shape:
        key, _, value = item.partition("=")
        if not _ or not value.isdigit():
            parser.error("--shape expects KEY=INTEGER")
        bindings[key] = int(value)
    source = args.path.read_text()
    report = build_source_map(source, shape_bindings=bindings or None)
    print(json.dumps(report, indent=2) if args.json else render_markdown(report, source))
    return 0


__all__ = ["build_source_map", "render_markdown", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
