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
