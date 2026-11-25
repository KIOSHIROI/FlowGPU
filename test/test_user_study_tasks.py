"""User-study fixtures are complete and cover guard + semantic stages."""
from benchmarks.audit.user_study import STAGES, TASKS, summary


def test_tasks_are_unique_and_complete():
    ids = [task["id"] for task in TASKS]
    assert len(ids) == len(set(ids)) == 6
    assert {task["stage"] for task in TASKS} == STAGES
    for task in TASKS:
        assert task["root_cause"] and task["evidence"]


def test_summary_counts_by_stage():
    data = summary()
    assert data["tasks"] == 6
    assert data["with_guard"] == 3
    assert sum(data["stages"].values()) == 6
