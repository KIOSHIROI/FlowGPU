# Benchmark evidence protocol

One page, mandatory for any number that appears in a paper, report, deck, or
submission message.  It was written after studying
`docs/EXTERNAL_STUDY_GPU_UNLOCK_ADA_20261027.md`, whose matched-pair discipline
is the standard this protocol copies (not its hardware work).

The rule: **a number is only as good as the comparison it came from.**  Every
claim answers four questions — what ran, against what, on which machine, and
with which repetitions.

## 1. Claim types (never mix them)

| Type | Example | What it may support | What it may NOT support |
|---|---|---|---|
| Instruction/issue rate | FP16/FP32 tensor TF/s | "this path is faster in isolation" | application speedup |
| Isolated kernel | GPU MODE matmul us | "this kernel is faster on this shape/GPU" | end-to-end model speedup |
| End-to-end workload | tokens/s, full model latency | user-facing claim | compiler-quality claim |
| Static/byte evidence | emitted-source SHA-256 | identity/contract claims | any performance claim |

Compiler-quality claims additionally require **generated vs extern vs hand**
provenance separation (`docs/SCORE_ATTRIBUTION_20261027.md`); a hand-written
extern kernel's score never counts as compiler output.

## 2. Matched-pair minimum (`docs/evidence/<topic>/README.md`)

A performance claim needs at least one pair that differs **only** in the thing
being tested:

- same physical GPU (record UUID or name), same driver/CUDA/torch/triton versions
  (record them; do not assume), same kernel binary/source hash, same launch
  configuration, same power/clock policy;
- stock/hand baseline measured **before** the change, with the same binaries;
- every repetition retained, not just the best; report median and spread;
- controls that should not move (an unrelated operator, a library baseline, a
  correctness check) and report them even when they are unchanged or slower;
- if a pair is impossible, say so and downgrade the claim to "single run,
  not a matched comparison".

## 3. Hash pins (copy the sources, not the smile)

Each evidence bundle records:

- commit/tag of this repo (`git describe --tags --always`), dirty-tree status;
- SHA-256 of generated modules/submission files (our `results/` already keeps
  candidates; pin the exact file);
- for GPU MODE: problem, leaderboard, GPU, mode, repetitions, and the exact
  CLI command; for CANN/FlagOS: submission id, endpoint, quota state, case
  count, exact pass set;
- platform-provided artifacts are linked by id, not rehosted.

## 4. Correctness gate before performance

- correctness must be checked on the exercised path (exact or tolerance-based as
  the task grades it), with mismatch counts reported;
- performance of an implementation that fails correctness is not recorded as a
  result — it goes in the failure table with why;
- prefer exact-reference checks when the task is integer/threshold logic (our
  packed-layout oracles are the model: independent implementation + byte compare).

## 5. Limitations paragraph (required)

Every report states: number of cards/runs, single-vs-multi boot, chip/driver
scope, power/clocks/cooling, reproducibility status, what was NOT tested, and
any control that behaved unexpectedly.  "One card, one cold boot" is a valid
result — overstating it is not.

## 6. Mapping to what FlowGPU already has

| Protocol item | Existing artifact | Gap |
|---|---|---|
| CPU correctness + byte identity | `results/pytest.xml`, `results/pytest.log`, golden/byte tests | fine |
| Generated vs hand provenance | `competition/attribution_20261027.json`, `docs/SCORE_ATTRIBUTION_20261027.md` | fine |
| GPU MODE matched pairs | `competition/gpu_mode/**` submission records | **needs** driver/GPU/repetition/power pins per entry |
| CANN/FlagOS official evidence | submission ids, artifacts, quotas in work logs | add explicit limitations per claim |
| Device execution of generated code | nightly `tools/gpu_smoke.py` | needs a registered runner before it is evidence |
| External-methodology audit | this page + study note | apply on the next GPU MODE submission |

## 7. Reporting template

```text
Claim: <one sentence, with the metric and the direction>
Comparison: <baseline> vs <candidate>; only <variable> differs
Hardware: <GPU>, <driver>, <CUDA>, <torch/triton>, power cap <W>, clocks <MHz>
Artifacts: repo <tag/commit>, module sha256 <hash>, submission <id>
Correctness: <exact/tolerance>, mismatches <n>, reference <what>
Repetitions: <k runs>, values <all>, median <x>, spread <y>
Controls: <what stayed flat or slower>
Limitations: <cards, runs, power/thermal, untested scope>
Not claimed: <e.g. end-to-end speedup, compiler quality>
```
