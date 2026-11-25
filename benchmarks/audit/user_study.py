"""User-study task fixtures for the audit-trail experiment.

Six faults from the Phase-2 matrix. Each task has a reference root cause,
stage and the audit evidence that should shorten localization.
"""
from __future__ import annotations

import json

TASKS = (
    {
        "id": "shape",
        "source": "mlp (B=4, D=64, H=128)",
        "fault": "call x with shape [3,64]",
        "root_cause": "input shape does not match the bound B",
        "stage": "host guard",
        "evidence": "input x dtype or shape does not match its declaration",
        "keywords": ("shape", "dimension", "b"),
    },
    {
        "id": "dtype",
        "source": "mlp",
        "fault": "call x as fp32",
        "root_cause": "input dtype does not match bf16",
        "stage": "host guard",
        "evidence": "input x dtype or shape does not match its declaration",
        "keywords": ("dtype", "bf16", "float"),
    },
    {
        "id": "loc_range",
        "source": "masked decode block (32 slots)",
        "fault": "loc = 99",
        "root_cause": "cache index outside [0,32)",
        "stage": "host guard",
        "evidence": "loc index is outside the cache range",
        "keywords": ("loc", "range", "index", "cache"),
    },
    {
        "id": "mask_missing",
        "source": "masked decode block",
        "fault": "mask is all zeros instead of the written prefix",
        "root_cause": "mask does not match the written slots",
        "stage": "semantic / oracle",
        "evidence": "audit shows the masked-attention kernel and reference; no host guard",
        "keywords": ("mask", "prefix", "attention"),
    },
    {
        "id": "cache_skip",
        "source": "masked decode block",
        "fault": "loc jumps from 8 to 10",
        "root_cause": "cache position skipped; slot 9 stays empty",
        "stage": "semantic / oracle",
        "evidence": "reference output differs; loc is inside the guarded range",
        "keywords": ("cache", "position", "skip", "loc"),
    },
    {
        "id": "moe_tie",
        "source": "moe_router_topk",
        "fault": "ids differ while routing weights match",
        "root_cause": "top-k tie order: lower column wins",
        "stage": "semantic contract",
        "evidence": "topk_last kernel; stable-argsort reference",
        "keywords": ("tie", "topk", "column", "ids"),
    },
)

STAGES = {"host guard", "semantic / oracle", "semantic contract"}


def summary():
    return {
        "tasks": len(TASKS),
        "stages": {stage: sum(1 for task in TASKS if task["stage"] == stage)
                   for stage in sorted(STAGES)},
        "with_guard": sum(1 for task in TASKS if task["stage"] == "host guard"),
    }


def main() -> int:
    print(json.dumps({"tasks": TASKS, "summary": summary()}, indent=2))
    return 0


__all__ = ["TASKS", "STAGES", "summary"]


if __name__ == "__main__":
    raise SystemExit(main())
