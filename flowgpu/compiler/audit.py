"""Build a human- and machine-readable audit trail for one pipeline.

The report is deliberately derived from the compiled artifacts, not from the
source text: kernels, fusion groups, host guards and the independent reference
are what an auditor can actually check.
"""
from __future__ import annotations

import json

from flowgpu.compiler.compile import compile_source


def build_audit_report(source, *, shape_bindings=None, source_path=None):
    """Return a JSON-serialisable audit report for the first pipeline."""
    result = compile_source(source, source_path=source_path,
                            shape_bindings=shape_bindings)
    return audit_from_result(result)


def audit_from_result(result):
    """Build the report from an already compiled result (no recompilation)."""
    program = result.pipelines[0]
    kernels = [{
        "name": kernel.name,
        "kind": kernel.kind,
        "ops": list(kernel.ops),
        "tile": list(kernel.tile) if kernel.tile else None,
        "grid": kernel.grid,
        "parameters": list(kernel.parameters),
    } for kernel in program.triton.kernels]
    groups = [{
        "anchor": group.anchor.op if group.anchor is not None else "pointwise",
        "members": [node.op for node in group.members],
        "fused_values": list(group.fused_values),
        "epilogue": [node.op for node in group.epilogue],
    } for group in program.fusion.groups]
    guards = [line.strip() for line in program.triton.host.splitlines()
              if "raise ValueError" in line]
    reference = [{
        "name": kernel.name,
        "kind": kernel.kind,
        "ops": list(kernel.ops),
        "results": list(kernel.results),
    } for kernel in program.reference.kernels]
    inputs = [{
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": [str(axis) for axis in tensor.shape],
    } for name, tensor in program.reference.inputs]
    names = {entry["name"] for entry in inputs}
    risks = []
    if any("mask" in name for name in names):
        risks.append("mask/cache consistency is not checked by the host")
    if "loc" in names and any("cache" in name for name in names):
        risks.append("loc range is guarded; cache-position continuity is not")
    if "valid" in names:
        risks.append("valid must match the written prefix; it is caller-owned")
    if any("q_scale" in name for name in names):
        risks.append("q_scale value is caller-owned; it is not checked")
    if any("cache" in name for name in names):
        risks.append("cache values are caller-owned; corruption is not checked")
    if any("topk" in op for kernel in kernels for op in kernel["ops"]):
        risks.append("topk tie contract: lower column wins")
    return {
        "pipeline": program.pipeline.name,
        "inputs": inputs,
        "semantic_risks": risks,
        "kernels": kernels,
        "fusion_groups": groups,
        "host_guards": guards,
        "reference_kernels": reference,
        "refusals": [refusal.reason for refusal in program.triton.refusals],
        "skipped": [entry.message() for entry in program.triton.skipped],
        "kernel_count": len(kernels),
        "reference_count": len(reference),
    }


def render_markdown(report) -> str:
    """Render the report as a short audit sheet."""
    lines = [f"# Audit: {report['pipeline']}", "",
             f"- kernels: {report['kernel_count']}",
             f"- reference kernels: {report['reference_count']}",
             f"- fusion groups: {len(report['fusion_groups'])}",
             f"- host guards: {len(report['host_guards'])}", ""]
    lines.append("## Inputs")
    for value in report["inputs"]:
        lines.append(f"- `{value['name']}` {value['dtype']} {value['shape']}")
    if report["semantic_risks"]:
        lines.append("\n## Semantic risks (caller-owned)")
        lines.extend(f"- {risk}" for risk in report["semantic_risks"])
    lines.append("## Kernels")
    for kernel in report["kernels"]:
        lines.append(f"- `{kernel['name']}` {kernel['kind']} "
                     f"{' -> '.join(kernel['ops'])}"
                     f"{' tile=' + str(kernel['tile']) if kernel['tile'] else ''}")
    lines.append("\n## Fusion")
    for group in report["fusion_groups"]:
        lines.append(f"- anchor `{group['anchor']}`: "
                     f"{' -> '.join(group['members'])}"
                     f" fused={group['fused_values']}")
    lines.append("\n## Host guards")
    for guard in report["host_guards"][:20]:
        lines.append(f"- {guard}")
    if len(report["host_guards"]) > 20:
        lines.append(f"- ... {len(report['host_guards']) - 20} more")
    lines.append("\n## Reference")
    for kernel in report["reference_kernels"]:
        lines.append(f"- `{kernel['name']}` {kernel['kind']} "
                     f"{' -> '.join(kernel['ops'])} -> {kernel['results']}")
    if report["refusals"]:
        lines.append("\n## Refusals")
        lines.extend(f"- {reason}" for reason in report["refusals"])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    import argparse
    import pathlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=pathlib.Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_audit_report(args.path.read_text(), source_path=str(args.path))
    print(json.dumps(report, indent=2) if args.json else render_markdown(report), end="")
    return 0


__all__ = ["build_audit_report", "audit_from_result", "render_markdown", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
