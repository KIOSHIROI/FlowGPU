"""The audit sheet exposes evidence for all six user-study faults."""
from benchmarks.audit.audit_coverage import main


def test_coverage_runs(capsys):
    assert main() == 0
    text = capsys.readouterr().out
    assert '"visible": 6' in text
    assert '"direct_visible": 3' in text
    assert '"risk_flagged": 3' in text
