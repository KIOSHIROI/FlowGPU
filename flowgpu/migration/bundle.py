"""Evidence-bundle validation and rendering.

A bundle is the product deliverable: one model, a set of backend profiles, and
for every (operator, platform) either an evidence record or an explicit gap.
The validator enforces the evidence protocol:
correctness needs a reference and a mismatch count; a pass needs an attached
artifact whose SHA-256 matches; performance needs a baseline and repetitions;
attribution is explicit; every scoped pair is accounted for.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

SCHEMA = "flowgpu.migration.bundle/1"
IMPLEMENTATIONS = {"generated", "extern", "hand"}
CORRECTNESS = {"pass", "fail", "not-run"}
MODES = {"live", "recorded"}
CORRECTNESS_MODES = {"exact", "tolerance"}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path) -> dict:
    return json.loads(pathlib.Path(path).read_text())


def validate(bundle: dict, base_dir) -> list:
    """Return a list of human-readable problems; empty means valid."""
    base = pathlib.Path(base_dir)
    problems = []
    if bundle.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA!r}")
    for key in ("model", "scope", "operators", "evidence", "gaps", "reproduction"):
        if key not in bundle:
            problems.append(f"missing key {key!r}")
    scope = bundle.get("scope") or {}
    platforms = scope.get("platforms") or []
    mode = scope.get("mode")
    if mode not in MODES:
        problems.append(f"scope.mode must be one of {sorted(MODES)}")
    if not platforms:
        problems.append("scope.platforms must not be empty")
    operator_names = {entry["name"] for entry in bundle.get("operators", [])}
    pairs = {(name, platform) for name in operator_names for platform in platforms}

    seen = set()
    for record in bundle.get("evidence", []):
        operator, platform = record.get("operator"), record.get("platform")
        pairs_key = (operator, platform)
        if operator not in operator_names:
            problems.append(f"evidence for unknown operator {operator!r}")
        if platform not in platforms:
            problems.append(f"evidence for out-of-scope platform {platform!r}")
        if pairs_key in seen:
            problems.append(f"duplicate evidence for {operator} on {platform}")
        seen.add(pairs_key)
        implementation = record.get("implementation")
        if implementation not in IMPLEMENTATIONS:
            problems.append(f"{operator}/{platform}: implementation must be one of {sorted(IMPLEMENTATIONS)}")
        correctness = record.get("correctness") or {}
        status = correctness.get("status")
        if status not in CORRECTNESS:
            problems.append(f"{operator}/{platform}: correctness.status must be one of {sorted(CORRECTNESS)}")
        if status in {"pass", "fail"}:
            if correctness.get("mode") not in CORRECTNESS_MODES:
                problems.append(f"{operator}/{platform}: correctness.mode must be one of {sorted(CORRECTNESS_MODES)}")
            if not correctness.get("reference"):
                problems.append(f"{operator}/{platform}: correctness needs a reference")
            if not isinstance(correctness.get("mismatches"), int):
                problems.append(f"{operator}/{platform}: correctness needs an integer mismatch count")
            elif status == "pass" and correctness["mismatches"] != 0 and correctness.get("mode") == "exact":
                problems.append(f"{operator}/{platform}: an exact pass cannot have mismatches")
        performance = record.get("performance")
        if performance:
            for field in ("metric", "unit", "value", "baseline"):
                if performance.get(field) in (None, ""):
                    problems.append(f"{operator}/{platform}: performance needs {field!r}")
            if not performance.get("repetitions"):
                problems.append(f"{operator}/{platform}: performance needs repetitions")
        artifacts = record.get("artifacts") or []
        if status == "pass" and not artifacts:
            problems.append(f"{operator}/{platform}: a pass needs at least one artifact")
        for artifact in artifacts:
            path = base / artifact.get("path", "")
            if not path.exists():
                problems.append(f"{operator}/{platform}: artifact {artifact.get('path')!r} is missing")
                continue
            if artifact.get("sha256") != _sha256(path):
                problems.append(f"{operator}/{platform}: artifact {artifact.get('path')!r} sha256 mismatch")
        if status == "pass" and not record.get("limitations"):
            problems.append(f"{operator}/{platform}: a pass needs a limitations statement")

    evidence_keys = seen
    gaps = {}
    for gap in bundle.get("gaps", []):
        key = (gap.get("operator"), gap.get("platform"))
        if key in evidence_keys:
            problems.append(f"gap duplicates evidence for {key[0]} on {key[1]}")
        if gap.get("operator") not in operator_names or gap.get("platform") not in platforms:
            problems.append(f"gap for out-of-scope pair {key}")
        if not gap.get("reason"):
            problems.append(f"gap {key} needs a reason")
        gaps[key] = gap
    missing = pairs - evidence_keys - set(gaps)
    for operator, platform in sorted(missing):
        problems.append(f"no evidence and no gap for {operator} on {platform}")

    acceptance = bundle.get("acceptance")
    if acceptance is not None:
        if not isinstance(acceptance, dict):
            problems.append("acceptance must be an object")
        else:
            if acceptance.get("correctness") == "required":
                for record in bundle.get("evidence", []):
                    if record.get("correctness", {}).get("status") != "pass":
                        problems.append(
                            "acceptance requires correctness: evidence for "
                            f"{record.get('operator')}/{record.get('platform')} is not a pass")
            floor = acceptance.get("performance_floor")
            if floor is not None:
                performance = [record for record in bundle.get("evidence", [])
                               if record.get("performance")]
                if not performance:
                    problems.append("acceptance performance_floor needs at least one performance record")
                for record in performance:
                    speedup = record["performance"].get("speedup")
                    if speedup is None:
                        problems.append(
                            "acceptance performance_floor needs a numeric speedup for "
                            f"{record['operator']}/{record['platform']}")
                    elif speedup < floor:
                        problems.append(
                            "acceptance performance_floor "
                            f"{floor} not met by {record['operator']}/{record['platform']} "
                            f"({speedup})")

    if bundle.get("reproduction") is not None and not bundle.get("reproduction"):
        problems.append("reproduction must list at least one command")
    return problems


def attribution_summary(bundle: dict) -> dict:
    summary = {"generated": 0, "extern": 0, "hand": 0}
    for record in bundle.get("evidence", []):
        summary[record.get("implementation")] += 1
    return summary


def render_one_page(bundle: dict) -> str:
    """A decision page: verdict, thresholds, owners, top gaps, next action.

    This is the page a CTO/procurement reviewer reads; the full evidence,
    coverage and reproduction commands stay in the long report.
    """
    evidence = bundle.get("evidence", [])
    gaps = bundle.get("gaps", [])
    platform_list = bundle["scope"]["platforms"]
    passed = [record for record in evidence if record["correctness"]["status"] == "pass"]
    failed = [record for record in evidence if record["correctness"]["status"] != "pass"]
    accepted = (not gaps) and bool(evidence) and not failed
    acceptance = bundle.get("acceptance") or {}
    owner = bundle.get("owner", "-")
    reviewer = bundle.get("reviewer", "-")
    lines = [
        f"# Acceptance decision: {bundle['model'].get('name', bundle['model'])}",
        "",
        f"| | |",
        "|---|---|",
        f"| **Overall** | **{'ACCEPTED' if accepted else 'REJECTED'}** |",
        f"| Scope | {bundle['scope']['mode']}; platforms: {', '.join(platform_list)} |",
        f"| Evidence | {len(passed)} pass / {len(failed)} fail / {len(gaps)} explicit gaps |",
        f"| Threshold | correctness={acceptance.get('correctness', 'recorded')}, "
        f"performance_floor={acceptance.get('performance_floor', '-')} |",
        f"| Owner | {owner} |",
        f"| Reviewer | {reviewer} |",
        f"| Repo | {bundle.get('repo', {}).get('tag', '?')} "
        f"({bundle.get('repo', {}).get('commit', '?')}) |",
        "",
        "## Decision rule",
        "",
        "- **ACCEPTED** means every scoped (operator, platform) pair has passing "
        "evidence and there are no gaps; it may go to the next stage (integration, "
        "procurement, or production gate).",
        "- **REJECTED** means at least one scoped pair is a recorded failure or an "
        "explicit gap; it must not be reported as accepted, and the gaps below are "
        "the work list.",
        "",
    ]
    if gaps:
        lines += ["## Top gaps (the work list)", "", "| operator | platform | reason |",
                  "|---|---|---|"]
        for gap in gaps[:8]:
            lines.append(f"| `{gap['operator']}` | {gap['platform']} | {gap['reason']} |")
        if len(gaps) > 8:
            lines.append(f"| ... | ... | {len(gaps) - 8} more in the full report |")
        lines.append("")
    if failed:
        lines += ["## Recorded failures", "", "| operator | platform | mismatches |",
                  "|---|---|---:|"]
        for record in failed:
            lines.append(f"| `{record['operator']}` | {record['platform']} | "
                         f"{record['correctness'].get('mismatches', '?')} |")
        lines.append("")
    if passed:
        lines += ["## What is already proven", "", "| operator | platform | implementation |",
                  "|---|---|---|"]
        for record in passed:
            lines.append(f"| `{record['operator']}` | {record['platform']} | "
                         f"{record['implementation']} |")
        lines += ["", "Attribution counts generated/extern/hand separately; a hand-written "
                  "kernel is not compiler evidence.", ""]
    lines += ["## Next action", "",
              "- If rejected: pick the top gaps, decide build-vs-extern for each, and "
              "re-run the bundle with the new evidence.",
              "- If accepted: attach the full report and the reproduction commands to "
              "the acceptance record.", ""]
    return "\n".join(lines)


def render(bundle: dict) -> str:
    platforms = bundle["scope"]["platforms"]
    lines = [
        f"# Migration evidence: {bundle['model'].get('name', bundle['model'])}",
        "",
        f"- mode: **{bundle['scope']['mode']}**",
        f"- platforms: {', '.join(platforms)}",
        f"- repo: {bundle.get('repo', {}).get('tag', '?')} ({bundle.get('repo', {}).get('commit', '?')})",
        "",
    ]
    evidence_records = bundle.get("evidence", [])
    gaps = bundle.get("gaps", [])
    accepted = (not gaps) and bool(evidence_records) and all(
        record["correctness"]["status"] == "pass" for record in evidence_records)
    lines += ["## Verdict", "",
              f"**Overall: {'ACCEPTED' if accepted else 'REJECTED'}** "
              f"({sum(1 for r in evidence_records if r['correctness']['status'] == 'pass')} pass, "
              f"{sum(1 for r in evidence_records if r['correctness']['status'] != 'pass')} fail, "
              f"{len(gaps)} explicit gaps)", ""]
    acceptance = bundle.get("acceptance")
    if acceptance:
        lines += [f"Acceptance: correctness={acceptance.get('correctness', '-')}, "
                  f"performance_floor={acceptance.get('performance_floor', '-')}", ""]
    lines += [
        "## Coverage",
        "",
        "| operator | " + " | ".join(platforms) + " |",
        "|---|" + "---|" * len(platforms),
    ]
    evidence_by_pair = {(record["operator"], record["platform"]): record
                        for record in bundle.get("evidence", [])}
    gaps_by_pair = {(gap["operator"], gap["platform"]): gap for gap in bundle.get("gaps", [])}
    for entry in bundle["operators"]:
        cells = []
        for platform in platforms:
            record = evidence_by_pair.get((entry["name"], platform))
            if record:
                status = record["correctness"]["status"]
                cells.append(("✅" if status == "pass" else "❌") + f" {record['implementation']}")
            elif (entry["name"], platform) in gaps_by_pair:
                cells.append("⛔ gap")
            else:
                cells.append("—")
        lines.append("| `%s` | %s |" % (entry["name"], " | ".join(cells)))
    lines += ["", "## Evidence", "",
              "| operator | platform | impl | correctness | perf | limitations |",
              "|---|---|---|---|---|---|"]
    for record in bundle.get("evidence", []):
        correctness = record["correctness"]
        perf = "-"
        performance = record.get("performance")
        if performance:
            perf = f"{performance['value']} {performance['unit']} vs {performance['baseline']}"
        lines.append("| `%s` | %s | %s | %s (%s) | %s | %s |" % (
            record["operator"], record["platform"], record["implementation"],
            correctness["status"], correctness.get("mode", "-"), perf,
            record.get("limitations", "-")))
    gaps = bundle.get("gaps", [])
    if gaps:
        lines += ["", "## Gaps (explicit, not hidden)", "",
                  "| operator | platform | reason |", "|---|---|---|"]
        for gap in gaps:
            lines.append("| `%s` | %s | %s |" % (gap["operator"], gap["platform"], gap["reason"]))
    summary = attribution_summary(bundle)
    lines += ["", "## Attribution", "",
              f"generated **{summary['generated']}**, extern **{summary['extern']}**, "
              f"hand **{summary['hand']}** — a hand-written kernel is not compiler evidence.",
              "", "## Reproduction", ""]
    lines += [f"```sh\n{command}\n```" for command in bundle["reproduction"]]
    lines.append("")
    return "\n".join(lines)
