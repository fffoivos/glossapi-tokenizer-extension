# reports/ — index

All artifacts for review of the 04 Vanilla CPT 5 B run. Bullet-style index;
follow paths into each file for detail.

## Canonical reports

- `5B_REPORT.md` — the endpoint report. 14 sections covering executive summary, method, headline results, trajectory analysis with V4 v3 CIs, per-task structure, Plutus discussion, retention, BPB, compute, caveats, Task-2 implications, 10 B stretch, tasks completed, references. **This is the report.**
- `decisions_matrix_20260529.md` — 24-row decisions matrix (A–X). Each row: issue, severity, plan ref, Apertus-paper ref, recommendation, action, status. Cross-checked across the three substantive investigations (V4 CIs, matched-config base eval design, script audit).
- `plutus_investigation_20260530.md` — the focused investigation of the Plutus QA 0.4889 → 0.4356 drop iter 834 → 1192. Marginal + paired bootstrap CIs, item-level diff (26 regressions / 14 gains), McNemar exact test (p=0.081), bakeoff Plutus comparison. Verdict: ambiguous, leaning small-real.
- `script_audit_20260529.md` — comprehensive shell-quoting / Slurm-export audit. 3 Critical + 7 Major + 9 Minor findings. Triggered by the `--export` comma-bug discovery that silently broke iter 238 + iter 477 native MCQ. The bug itself is fixed; deferred items are tracked here.
- `gpu_hours_breakdown_20260530.md` — full per-job + per-category GPU-h accounting. Final total 217.2 GPU-h. Methodology footer + adversarial-verify section.
- `config_geometry_audit_iter_0000119.{md,json}` — the historical iter-119 audit that first documented the RoPE/seqlen mismatch (Vanilla-0.5B Critical-1). Captures the THEN-state of hyperparameters.json (which had the contradiction). Preserved as the historical record; hyperparameters.json has since been corrected.

## V4 bootstrap CIs

- `v4_bootstrap_cis_native_mcq.json` — current v3 revision. 10 models (Apertus-Base Path A + matched-config Path-B-perturbed + bakeoff Vanilla 2/3.5/5B + iter 119/238/477/834/1192). 83 delta_table rows covering 5 metrics × the load-bearing paired comparisons. Methodology: 1000 resamples, percentile, 95%, rng_seed=20260529, per-task item-level then macro-mean across the 3 headline tasks per resample. Paired bootstrap indices for delta CIs.

## Plots

- `plot_trajectory_native_mcq_v1.py` + `.png` — overall 3-task headline trajectory: 04-run + bakeoff + Apertus-Base Path A CI band + matched-config init floor. 04-run CIs from V4 v2 on the first 4 points; iter 834 + 1192 marked as diamonds (CIs pending a V4 v3 plot refresh).
- `plot_mmlu_trajectory.py` + `.png` — GreekMMLU (left) vs English MMLU (right) trajectories. GreekMMLU monotone climb 0.4985 → 0.5584 with V4 v3 CIs; English MMLU non-monotonic with approximate ±1.96·stderr ribbons.
- `plot_lm_loss_trajectory.py` + `.png` — training lm loss vs consumed tokens, raw + EMA-smoothed, with checkpoint markers + warmup-end vertical line. Shows the warmup descent → post-warmup slow-grind shape.
- `plot_retention_per_language.py` + `.png` — EN / FR / DE / RU retention per language (global_mmlu macro-mean + xnli). Reference dotted line = matched-config Apertus-Base = our true pre-CPT init. Caption documents the Δ vs true init per task per language.

## Per-checkpoint sidecar handoff records

- `iter_<N>_checkpoint_sidecar_handoff_pass.json` and `iter_<N>_checkpoint_sidecar_verify_latest.json` — per-checkpoint verification snapshots. The `_handoff_pass` file records the moment handoff_ready=true was reached; the `_verify_latest` file is the latest verifier output (sometimes the same content, sometimes more recent). Audit trail for the watcher chain.
- `iter_0000119_checkpoint_sidecar_precheck.json` — a one-off precheck record from the iter-119 sidecar repair (see RUN_LOG §"Iter 119 Checkpoint Handoff And Conversion Repair").

## Status renderer state

- `latest_5b_report_state.json` — last collected state JSON. Includes canonical-task filtered metrics + decision_state + checkpoint state. Updated by `scripts/collect_5b_report_state.py`.
- `latest_5b_report_status.md` — the human-readable status snapshot. Rendered from the state JSON by `scripts/render_5b_report_status.py`. Filtered through `goal/canonical_eval_tasks.json` so only canonical tasks appear.

## Local caches (kept for re-rendering)

- `eval_data_cache_5b/` — per-checkpoint copies of native MCQ + retention + BPB summary JSONs. 2.2 MB. Used by `plot_mmlu_trajectory.py` + `plot_retention_per_language.py` to avoid re-pulling from Clariden on every plot regen.
- `train_logs_cache_5b/` — copies of the training log files per chain segment. 556 KB. Used by `plot_lm_loss_trajectory.py`.

## Audit & methodology notes

- See `script_audit_20260529.md` for the comprehensive script audit.
- See `decisions_matrix_20260529.md` rows N + O for the iter-119 handoff repair history (first conversion job failed because mirrored config was missing).
- See `gpu_hours_breakdown_20260530.md` for the AllocTRES caveat — every GPU job in this cohort requested gres/gpu=4 explicitly, so the whole-node billing assumption was not actually exercised (would matter for any future `--gpus-per-node=1` job).
