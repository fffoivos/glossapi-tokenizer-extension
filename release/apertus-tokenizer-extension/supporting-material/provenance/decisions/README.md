# Decisions provenance

Plans + plan-vs-results reconciliation + master synthesis for the Apertus Greek CPT 4-arm bakeoff.

Read order if you're new:

1. **`CPT_MASTER_20260526.md`** — single-doc synthesis (mission, plans, fidelity, results, discrepancies, what's needed for production). The canonical reference.
2. **`PLAN_VS_RESULTS_RECONCILIATION_20260526.md`** — reads the bakeoff result back into the v0.12 experimental-design plan; §10 is a 14-entry discrepancy log.

Both docs are mirrored from the GitHub source repo:

```text
https://github.com/fffoivos/glossapi-tokenizer-extension/tree/main/subprojects/03_apertus_extension_and_embedding_adaptation
```

The GitHub copies live at the top level of stage 03 (`CPT_MASTER_20260526.md`) and under `_archive/synthesis_sources_20260526/` (the reconciliation, since it was synthesized into CPT_MASTER and archived alongside the other source docs).

## Cross-references

- Benchmark results: `../../../benchmark-evals/bakeoff-final/` (5B headline) and `../../../benchmark-evals/3.5B-comparison/` (iso-token snapshot).
- Per-stage eval narratives: `../evals/BAKEOFF_FINAL_RESULTS_20260526.md` (5B endpoint), `../evals/CONTINUATION_3P5B_RESULTS_20260525.md` (3.5B endpoint), `../evals/V4_BENCHMARK_COMPARISON.md` (baseline).
- TD specifics: `../token-distillation/TOKEN_DISTILLATION_PLAN.md`, `../token-distillation/RUN_LOG_20260523.md`.
- Tokenizer selection: `../tokenizer-selection/CHOSEN_CUTOFF.md` and `../tokenizer-selection/CUTOFF_SWEEP_REPORT.md`.

## The headline result is not rule-bound

Both decision docs flag the same load-bearing caveat: the pre-commit decision-rule thresholds from `old_experiments_plan.md` v0.12 §10 Q8 (X / M_progress / M_ext / M_van / T) were never locked before bakeoff results came in. The 5B headline ("TokenDistil-5B wins downstream, Vanilla-5B wins BPB") is therefore an honest description of the numbers, not an adjudicated v0.12-§10-Q8 winner.

The full discrepancy log (14 entries, 6 HIGH-severity including this one) is in `PLAN_VS_RESULTS_RECONCILIATION_20260526.md` §10.
