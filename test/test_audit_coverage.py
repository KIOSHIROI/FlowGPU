"""The audit sheet exposes evidence for all six user-study faults."""
from benchmarks.audit.audit_coverage import main


def test_coverage_runs(capsys):
    assert main() == 0
    text = capsys.readouterr().out
    assert '"visible": 4' in text
    assert '"semantic_visible": 1' in text
