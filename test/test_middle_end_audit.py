"""Middle-end opportunity audit tool smoke test."""
from __future__ import annotations

import json

from tools import middle_end_opportunity_audit


def test_audit_runs_on_one_program_and_reports_ablation_deltas(tmp_path):
    out = tmp_path / "audit.json"
    assert middle_end_opportunity_audit.main(
        ["--only", "pointwise_chain", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert data["totals"]["programs"] == 1
    row = data["rows"][0]
    assert row["default"]["kernels"] == 1
    assert "rule_fusion" in row["deltas"]
    assert row["deltas"]["rule_fusion"]["kernel_delta"] == 0
    assert (tmp_path / "audit.md").exists()
