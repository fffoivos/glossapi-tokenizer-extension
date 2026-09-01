# greekmmlu_trajectory_20260822 — matched cross-scale trajectory report

> **In one line:** the 34-checkpoint matched 1.5B-vs-8B GreekMMLU trajectory that answered the scale-mirroring question with a clear **no** — and, in the same document, flagged that its own question panel had been decontaminated twice.
> **Period:** 2026-08-22 (`de52e1bc`). **Status:** superseded as the headline by [`../hard_h2g_full_panel_stable_lr_20260822/`](../hard_h2g_full_panel_stable_lr_20260822/), which moved the primary metric to the full 16,632-question public panel; its cross-scale conclusion still stands.
> **Narrative:** [`../../HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md`](../../HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md).

## Why this existed

Goal B of the study was whether a 1.5B model, trained on identical data in identical order, reproduces the *shape* of the 8B GreekMMLU trajectory well enough to serve as a cheap screening proxy. Answering that needs every checkpoint of both runs scored on one frozen panel with one evaluator — 17 checkpoints per scale, `17 × 2 × 16,159 = 549,406` checkpoint-question observations.

## What was run

All 34 checkpoints were exported to HF and scored in a single 4-node / 16-GH200 allocation, Slurm `3148664`, state `COMPLETED`, elapsed `03:05:13` ≈ 49.4 allocated GPU-hours: 32 new results plus two reused receipt-bound ones; 31 new exports plus three reused. The frozen evaluator bundle is tree SHA-256 `b7a9e144…`, the aggregate `49c43d10…`. Source roots for every checkpoint are in [`evidence/checkpoint_sources.tsv`](evidence/checkpoint_sources.tsv).

## Result

1.5B does not mirror 8B. Across the HPLT boundary (update 2,261) to the OpenArchives endpoint (3,218) the two scales move in *opposite* directions: −3.09 pp at 1.5B, +1.27 pp at 8B. Accuracy-level Pearson correlation is −0.6698; adjacent-checkpoint change correlations are −0.0581 (Pearson) and 0.1265 (Spearman). Of 31 subjects, 29 peak during HPLT at 1.5B while 22 peak during OpenArchives at 8B ([`evidence/analysis_summary.json`](evidence/analysis_summary.json) records 22 of 29 subjects ending below their first measurement at 1.5B, and 29 of 29 ending above at 8B). The qualification that matters: 1.5B's final choice NLL (1.34670) is *better* than its first measured value (1.38264) even though its accuracy fell — probability quality and argmax accuracy came apart.

## Caveats the report itself carries

- Five of the 34 exports (1.5B updates 238, 952, 1,428, 2,856 and 8B update 714) missed the stricter cross-runtime logit threshold and are receipted as scoped to this evaluator only — see [`evidence/export_parity_audit.json`](evidence/export_parity_audit.json). Very small checkpoint differences must not be over-read.
- The historical curve drawn for context is the **earlier surviving two-arm TD study** (≈58.75% peak, ≈58.66% final), not the selected β₂ arm that is the formal replication target.
- Update 0 was not evaluated on this panel, and the online Old-Greek panel is excluded from retention claims as unreliable.
- The 16,159-question panel removes 473 GreekMMLU questions that the *training streams* had already been decontaminated against — a second application of decontamination. This is why the full public panel became the corrected primary a day later.

## Contents

`GREEKMMLU_H2G_CROSS_SCALE_TRAJECTORIES_20260822.html`, `build_report.py`, `analyze_report_evidence.py`, `verify_report.py`, `evidence/` (aggregate, analysis summary, checkpoint sources, export parity audit, per-checkpoint export receipts, training trajectories, the replication contract, the historical two-arm curve) and `qa/` with its passing receipt. Rebuild instructions are in the handoff document, §10.
