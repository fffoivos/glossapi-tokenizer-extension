# Hard H2G full-panel + no-decay execution notes for review

Date: 2026-08-22

Binding plan:
[HARD_H2G_FULL_PANEL_SCORING_AND_NO_DECAY_BRANCH_PLAN_20260822.md](HARD_H2G_FULL_PANEL_SCORING_AND_NO_DECAY_BRANCH_PLAN_20260822.md)

Status: **execution in progress**

This is the append-only reviewer-facing operational record. It separates
scientific decisions, allocation-free checks, scheduler events, execution
receipts, failures, and any deviation from the plan. Commands are recorded in
sanitized form; credentials and private infrastructure tokens are excluded.

## Frozen owner decisions

- Apply Track A full-public 8B scoring and Track B no-decay 8B branch.
- Full 16,632-question GreekMMLU is the corrected primary evaluation.
- Existing 16,159-question results remain a secondary sensitivity analysis.
- The formal historical target is the selected beta2=0.999 arm
  `curr_td_b20p999_b3p999_13b_20260616T093527Z`, not the earlier surviving
  two-arm visualization curve.
- Replication band ratified before A2 results: best legacy-BF16 accuracy must
  be within plus or minus 1.0 percentage point of `9973/16632 = 59.9627%`.
- 1.5B peak-LR search remains deferred.
- B3 is conditional on the measured no-decay trajectory through update 3,218.
- No dataset rebuild, re-deduplication, or anonymization audit is in scope.

## Event log

### 2026-08-22 19:14–19:20 Europe/Athens — intake

- Read the binding plan in full.
- Read the installed `prepare-apertus-experiment` and
  `optimize-apertus-cscs` skills and their launch, benchmark, runner, and live
  cluster references.
- Verified local CSCS certificate validity through 2026-08-23 12:55:26.
- Verified read-only SSH access to `clariden`; no jobs were present for user
  `fffoivos` at 19:16:44.
- Fetched canonical `fffoivos/apertus-cscs-efficiency`; current
  `origin/main` is `ef69de358f8fe2c84d4842be074f117a9199f4df`.
- Read canonical `docs/REPOSITORY_SCOPE.md` and
  `docs/CANONICAL_CAMPAIGN_RUNNER.md` before adding lifecycle code.
- No Slurm job or allocation had been requested at this point.

### 2026-08-22 19:28 Europe/Athens — first allocation

- Allocation-free remote inventory confirmed all 17 8B checkpoint sources,
  the full-state update-2,499 branch checkpoint, and all reusable HF exports.
- Requested a direct foreground `salloc --no-shell` for Track A rather than a
  fire-and-forget batch job.
- Slurm job `3151839`: `normal`, four nodes / 16 GH200 GPUs, four-hour limit,
  job name `h2g-full-public-score`.
- The allocation was granted immediately. Its controlling SSH/salloc session
  remains attached while the immutable scientific bundle and inputs are
  finalized.

## Deviations

None yet.

## Failures and reusable findings

None yet.

## Receipt index

To be populated as artifacts are produced.
