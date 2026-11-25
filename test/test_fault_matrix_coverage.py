"""All 13 fault-matrix entries are represented in the audit sheet."""
from benchmarks.audit.fault_matrix_coverage import main


def test_all_faults_visible(capsys):
    assert main() == 0
    text = capsys.readouterr().out
    assert '"visible": 13' in text
    assert '"direct": 6' in text
    assert '"risk": 7' in text
