"""Answer scoring separates root cause from stage localization."""
from benchmarks.audit.user_study import TASKS
from benchmarks.audit.user_study_runner import score_answer


def test_correct_answer_scores_both():
    task = next(task for task in TASKS if task["id"] == "loc_range")
    result = score_answer(task, answer="loc is outside the cache range",
                          stage="host guard")
    assert result["root_cause_keyword"] and result["stage_correct"] and result["both"]


def test_right_cause_wrong_stage_is_not_both():
    task = next(task for task in TASKS if task["id"] == "loc_range")
    result = score_answer(task, answer="loc out of range", stage="semantic / oracle")
    assert result["root_cause_keyword"] and not result["stage_correct"]
    assert not result["both"]


def test_study_session_summarises_responses():
    from benchmarks.audit.user_study_runner import StudySession
    session = StudySession("s1", "treatment")
    session.record("shape", answer="input shape mismatch", stage="host guard", seconds=12)
    session.record("loc_range", answer="loc range", stage="host guard", seconds=8)
    session.record("mask_missing", answer="mask prefix", stage="semantic / oracle", seconds=20)
    summary = session.summary()
    assert summary["tasks"] == 3
    assert summary["both_correct"] == 3
    assert summary["median_seconds"] == 12
