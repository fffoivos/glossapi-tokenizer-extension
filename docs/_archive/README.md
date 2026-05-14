# Archived docs (pre-C3-convergence)

Everything here describes work that is **done**: the dedup pipeline,
near-dedup redesign, HF dedup investigation, pipeline E2E verification,
stage-checklist verification, and builder/tokenizer-efficiency plumbing
that produced the inputs C3 trained on.

Read these only if you're doing historical reconstruction. They are
not load-bearing for the C3 cutoff decision or the implementation /
embedding-adaptation work that follows it.

Start here instead:

- `../C3_CONVERGENCE.md` — what's settled, what's open
- `../C3_CUTOFF_REPORT.md` — fertility sweep results
- `../ACTIVE_BACKLOG.md` — the live work list
- `../PROJECT_INDEX.md` — map of every doc + subproject

## Contents

- `BUILDER_TOKENIZER_EFFICIENCY_PLAN.md` — tokenizer-worker perf sweep
  plan (RAYON_NUM_THREADS, batch size). Settled.
- `HF_DEDUP_INVESTIGATION.md` — comparing the in-repo dedup against
  HF/DataTrove MinHash. Settled.
- `NEAR_DEDUP_MEMORY_FOOTPRINT_TODO.md` — memory reductions in the
  near-candidate hot path. Settled in repo code.
- `NEAR_DEDUP_REDESIGN_PLAN.md` — the redesign that produced the
  current near-candidate streaming shape. Settled.
- `PIPELINE_E2E_STAGE_CHAIN.md`
- `PIPELINE_E2E_VERIFICATION_PLAN.md`
- `PIPELINE_E2E_VERIFICATION_TODO.md`
- `PIPELINE_E2E_WORKER_RUN_REPORT_20260415.md`
- `PIPELINE_RECOVERY_AND_SCALE_PLAN.md`
- `PIPELINE_STAGE_PARALLELISM_REVIEW_20260415.md`
- `PIPELINE_STAGE_PROGRESS_REVIEW_20260415.md` — the family of E2E +
  pipeline-stage verification docs. Settled.
- `STAGE_VERIFICATION_CHECKLIST.md` — per-stage checklist for the
  pipeline E2E. Settled.
