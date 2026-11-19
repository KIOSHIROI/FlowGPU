"""Cross-backend operator migration audit (product CLI).

    python tools/migration_audit.py coverage --model models/qwen2_7b_decode.json \
        --platforms ascend,ascend-wide,kunlunxin
    python tools/migration_audit.py bundle new --model ... --platforms ... --out evidence/run
    python tools/migration_audit.py bundle validate evidence/run
    python tools/migration_audit.py bundle render evidence/run -o evidence/run/report.md
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from flowgpu.migration import coverage as coverage_module  # noqa: E402
from flowgpu.migration import pricing as pricing_module  # noqa: E402
from flowgpu.migration import bundle as bundle_module  # noqa: E402


def _git() -> dict:
    """Repo identity; a source tarball without .git can pass it via FLOWGPU_REPO_*."""
    def run(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT,
                                           text=True).strip()
        except Exception:  # noqa: BLE001 - git may be absent; the field is then unknown
            return "unknown"
    tag = os.environ.get("FLOWGPU_REPO_TAG") or run("describe", "--tags", "--always")
    commit = os.environ.get("FLOWGPU_REPO_COMMIT") or run("rev-parse", "HEAD")
    dirty = os.environ.get("FLOWGPU_REPO_DIRTY")
    dirty = bool(run("status", "--porcelain")) if dirty is None else dirty == "1"
    return {"tag": tag, "commit": commit, "dirty": dirty}


def _coverage(args) -> int:
    result = coverage_module.coverage(args.model, args.platforms.split(","))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(coverage_module.render_coverage(result))
    return 0


def _price(args) -> int:
    if args.package:
        print(json.dumps(pricing_module.quote(args.package), indent=2))
    elif args.person_days:
        print(json.dumps(pricing_module.custom_quote(args.person_days), indent=2))
    else:
        print("| package | person-days | price (CNY) | cost floor | margin | scope |")
        print("|---|---:|---:|---:|---:|---|")
        for record in pricing_module.price_menu():
            print("| %s | %d | %d | %d | %.0f%% | %s |" % (
                record["package"], record["person_days"], record["price"],
                record["cost_floor"], record["gross_margin"] * 100,
                record["scope"]))
    return 0


def _init(args) -> int:
    """Create a customer workspace: model profile, evidence skeleton, runbook."""
    import hashlib as _hashlib
    workspace = pathlib.Path(args.out)
    workspace.mkdir(parents=True, exist_ok=False)
    model_source = pathlib.Path(args.model)
    model = coverage_module.load_model(model_source)
    (workspace / "model.json").write_text(model_source.read_text())
    evidence = workspace / "evidence"
    evidence.mkdir()
    workspace_json = {
        "schema": "flowgpu.migration.workspace/1",
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model": model["model"],
        "model_sha256": _hashlib.sha256(model_source.read_bytes()).hexdigest(),
        "platforms": args.platforms.split(","),
        "owner": args.owner or "TBD",
        "reviewer": args.reviewer or "TBD",
    }
    (workspace / "workspace.json").write_text(json.dumps(workspace_json, indent=2) + "\n")
    (workspace / "README.md").write_text(
        "# Migration acceptance workspace\n\n"
        f"Model: {model['model']}\nPLATFORMS: {args.platforms}\n\n"
        "```sh\n"
        "python -m flowgpu.migration.cli coverage --model model.json "
        f"--platforms {args.platforms}\n"
        "python -m flowgpu.migration.cli bundle new --model model.json "
        f"--platforms {args.platforms} --out evidence/run-1\n"
        "python -m flowgpu.migration.cli bundle validate evidence/run-1\n"
        "python -m flowgpu.migration.cli bundle render evidence/run-1 --one-page\n"
        "```\n\n"
        "Fill every evidence record or declare it as a gap; a pass needs a reference, "
        "a mismatch count, a SHA-256 artifact and a limitations statement.\n")
    (workspace / "runbook.md").write_text(
        "# Acceptance runbook\n\n"
        "| step | owner | status |\n|---|---|---|\n"
        "| confirm operator inventory (trace, not manual) | | |\n"
        "| confirm thresholds (correctness, performance floor) | | |\n"
        "| collect evidence per operator/platform | | |\n"
        "| run validator | | |\n"
        "| review one-page decision | | |\n"
        "| sign acceptance / record gaps | | |\n")
    print(f"created workspace {workspace} ({len(model['operators'])} operators)")
    return 0


def _demo(args) -> int:
    """Print the product flow on a bundled model (no GPU required)."""
    model = pathlib.Path(__file__).resolve().parent / "data" / "qwen2_7b_decode.json"
    result = coverage_module.coverage(model, ["cuda", "ascend", "ascend-wide"])
    print(coverage_module.render_coverage(result))
    return 0


def _bundle_new(args) -> int:
    result = coverage_module.coverage(args.model, args.platforms.split(","))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = coverage_module.load_model(args.model)
    manifest = {
        "schema": bundle_module.SCHEMA,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "repo": _git(),
        "model": {"name": model["model"], "profile_sha256": hashlib.sha256(
            pathlib.Path(args.model).read_bytes()).hexdigest()},
        "scope": {"mode": "recorded", "platforms": args.platforms.split(",")},
        "operators": [{"name": entry["name"]} for entry in model["operators"]],
        "evidence": [],
        "gaps": [{"operator": row["operator"], "platform": row["platform"],
                  "reason": f"{row['status']}: {row['reason']}"}
                 for row in result["rows"] if row["status"] in {"refused", "missing"}],
        "reproduction": ["python tools/migration_audit.py coverage --model %s --platforms %s"
                         % (args.model, args.platforms)],
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(manifest['gaps'])} coverage gaps)")
    return 0


def _bundle_validate(args) -> int:
    base = pathlib.Path(args.bundle)
    data = bundle_module.load(base / "manifest.json")
    problems = bundle_module.validate(data, base)
    if problems:
        for problem in problems:
            print("problem:", problem)
        return 1
    print(f"bundle OK: {len(data['evidence'])} evidence records, {len(data.get('gaps', []))} explicit gaps")
    return 0


def _bundle_render(args) -> int:
    base = pathlib.Path(args.bundle)
    data = bundle_module.load(base / "manifest.json")
    problems = bundle_module.validate(data, base)
    if problems:
        for problem in problems:
            print("problem:", problem)
        return 1
    text = (bundle_module.render_one_page(data) if args.one_page
            else bundle_module.render(data))
    if args.output:
        pathlib.Path(args.output).write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("coverage")
    p.add_argument("--model", required=True)
    p.add_argument("--platforms", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_coverage)

    p = sub.add_parser("init")
    p.add_argument("--model", required=True)
    p.add_argument("--platforms", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--owner")
    p.add_argument("--reviewer")
    p.set_defaults(func=_init)

    p = sub.add_parser("demo")
    p.set_defaults(func=_demo)

    p = sub.add_parser("price")
    p.add_argument("--package", choices=sorted(pricing_module.PACKAGES))
    p.add_argument("--person-days", type=int)
    p.set_defaults(func=_price)

    p = sub.add_parser("bundle")
    subsub = p.add_subparsers(dest="bundle_command", required=True)
    q = subsub.add_parser("new")
    q.add_argument("--model", required=True)
    q.add_argument("--platforms", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(func=_bundle_new)
    q = subsub.add_parser("validate")
    q.add_argument("bundle")
    q.set_defaults(func=_bundle_validate)
    q = subsub.add_parser("render")
    q.add_argument("bundle")
    q.add_argument("-o", "--output")
    q.add_argument("--one-page", action="store_true",
                   help="emit only the decision page (verdict, owners, top gaps)")
    q.set_defaults(func=_bundle_render)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
