# User-study protocol (not yet run)

## Question

> Can engineers localize a non-standard LLM-subgraph fault faster with
> FlowGPU's audit trail than with source code and logs alone?

## Design

Within-subject, two conditions, counterbalanced order:

- **control**: `.fgpu` source + failing command/stack trace;
- **treatment**: same plus `flowgpu.compiler.audit` sheet and the CPU reference.

## Tasks

Six injected faults, one per task:

1. input shape mismatch;
2. dtype mismatch;
3. cache `loc` out of range;
4. mask missing / too short;
5. skipped/duplicate cache position;
6. MoE top-k tie mismatch.

Each task asks for: root cause, affected stage (compile / guard / kernel /
reference), and the next debugging action.

## Participants

8–12 engineers with compiler, CUDA or inference-performance experience.
No personal data collected beyond experience level; sessions recorded only as
anonymised timings and answers.

## Metrics

- time to correct root cause;
- localization accuracy (stage + file/line/guard);
- confidence (5-point Likert);
- perceived usefulness of each audit section.

## Analysis

- paired comparison per participant (Wilcoxon signed-rank);
- report median time and accuracy with confidence intervals;
- effect size (matched-pairs rank-biserial);
- qualitative coding of where the audit sheet helped or misled.

## Artifacts

- the six task fixtures;
- a scoring rubric for root cause and stage;
- the audit report generator (`flowgpu.compiler.audit`);
- a consent/withdrawal script.


## Task fixtures

The six tasks are now code, not prose:

```bash
python -m benchmarks.audit.user_study
```

Each fixture records the source, fault, reference root cause, expected stage
and the audit evidence. Three tasks are host-guard cases; three are semantic
(mask, skipped cache position, MoE tie).

The fixtures are ready; the experiment still needs participants.

## Status

Protocol ready; no participants have been run. This document is the prereg
skeleton, not a result.
