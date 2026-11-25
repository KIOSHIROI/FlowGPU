"""Score one user-study answer against a task fixture.

This is the grading half of the study; it does not run a session or collect
personal data.
"""
from __future__ import annotations

import argparse
import json

from benchmarks.audit.user_study import TASKS


def score_answer(task, *, answer, stage):
    text = answer.lower()
    keyword = any(word.lower() in text for word in task["keywords"])
    stage_ok = stage == task["stage"]
    return {
        "task": task["id"],
        "root_cause_keyword": keyword,
        "stage_correct": stage_ok,
        "both": keyword and stage_ok,
        "expected_stage": task["stage"],
        "answer": answer,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=[task["id"] for task in TASKS])
    parser.add_argument("--answer", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    task = next(task for task in TASKS if task["id"] == args.task)
    result = score_answer(task, answer=args.answer, stage=args.stage)
    print(json.dumps(result, indent=2) if args.json else result)
    return 0


__all__ = ["score_answer"]


if __name__ == "__main__":
    raise SystemExit(main())
