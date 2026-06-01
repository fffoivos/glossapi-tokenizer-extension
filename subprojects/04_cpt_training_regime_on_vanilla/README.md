# 04 — CPT Training Regime on Vanilla

Task-1 regime-diagnostic experiment. Vanilla Apertus-8B continued-pretrained
to 5 B tokens under the Apertus-faithful regime (LR 1.1e-5 with 1.2 B-token
warmup then constant; AdEMAMix β3=0.99; Goldfish k=h=50; rope_theta=500K
Path B inherited from bakeoff). Tests whether the bakeoff CPT regime caused
the native-Greek MCQ degradation that hit every bakeoff arm.

**Status:** complete. 5 B endpoint hit 2026-05-30. Regime hypothesis
statistically supported (V4 v3 paired CIs, all five load-bearing
comparisons exclude zero — see `reports/v4_bootstrap_cis_native_mcq.json`).

## Where things live

| Path | Purpose |
|---|---|
| `cpt-plan.md` | The experimental plan (v1.0). §2 = Task 1 (this run), §3 = Task 2 (extension, deferred), §3.4 Q3.4.10 = Path-A recommendation for Task 2. |
| `goal/` | Locked spec: `goal.md` (brief), `hyperparameters.json` (authoritative settings — Path A base + Path B `training_geometry` override + Task-2 Path-A recommendation), `canonical_eval_tasks.json` (eval task lockdown filtering the renderer + reviewers to the 12 canonical retention tasks + 3 Greek MCQ headline + Plutus diagnostic + 3 heldout BPB). |
| `scripts/` | All run scripts: dataset build + train chain + sidecar fan-out + verifier + matched-config base eval + status renderer + adversarial-review runner. Live in the Clariden mirror at `/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/04_cpt_training_regime_on_vanilla/scripts/`. |
| `reports/` | Everything for review: the 5 B report, decisions matrix, V4 v3 bootstrap CIs, plots, per-checkpoint sidecar verifies, audits, GPU-h accounting, Plutus investigation. See `reports/README.md` for the index. |
| `adversarial_reviews/` | One folder per checkpoint (Vanilla-0.5B / 1B / 2B / 3.5B / 5B). Each folder has `prompt.md`, `adversarial_critique.md`, `review_metadata.env`. Reviewer backend = Claude Code subagent (codex was offline during the run; metadata records this). |
| `RUN_LOG_20260528.md` | Append-only narrative log of the entire run, every decision and every error in order. 117 KB. The full audit trail. |
| `monitor_logs/` | Watcher-side log files from the in-flight sidecar / status / verify tmux sessions. |
| `_archive/` | Intermediate scratch: bootstrap computation workspaces (v4_workspaces) + superseded drafts (5B_REPORT_DRAFT.md). Preserved for reproducibility; not part of the active report set. |

## Read in this order

1. `goal/goal.md` — what the experiment is.
2. `goal/hyperparameters.json` — the authoritative settings (Path A architecture + Path B training override + Task-2 Path-A recommendation).
3. `cpt-plan.md` §1 + §2 — context and Task-1 spec.
4. `reports/5B_REPORT.md` — the endpoint report (results + caveats + Task-2 implications).
5. `reports/decisions_matrix_20260529.md` — every decision row A–X with severity, plan ref, paper ref, recommendation, action, status.
6. `reports/v4_bootstrap_cis_native_mcq.json` — 1000-resample 95% percentile CIs, 10 models, 83 delta rows. Authoritative for every cross-arm headline claim.
7. `reports/plot_*.png` — visual storyline.
8. `TASK2_HANDOFF.md` (top level, this folder) — the response to the planning agent. Lists what we learned, what we got wrong, and what Task 2 should inherit.
9. `adversarial_reviews/Vanilla-5B/adversarial_critique.md` — the endpoint adversarial view.
10. `RUN_LOG_20260528.md` — the full audit trail if you need to know what happened on a specific date.

## Headline result (post-V4 v3, 3-task native Greek MCQ)

iter 1192 marginal headline = **0.4973 [0.4779, 0.5156]**. Compared to:

- bakeoff Vanilla-5B (matched tokens, Path B): Δ = **+6.69 pp**, CI [+5.13, +8.30] — significant.
- Apertus-Base Path A (released geometry): Δ = **+1.56 pp**, CI [+0.16, +2.84] — significant (barely).
- Matched-config Apertus-Base = our actual init: Δ = **+7.01 pp**, CI [+5.37, +8.57] — significant.
- iter 477 (post-warmup baseline, 2 B): Δ = +1.82 pp, CI [+0.60, +3.06] — significant.

Trajectory shape: warmup → +3.05 pp burst at iter 477 → flat 1.5 B-token segment → +1.84 pp endpoint lift at iter 1192. **Not** the bakeoff Vanilla "peak-early-then-drift" shape.

## Caveats (in priority order)

1. **Path A vs Path B geometry confound.** Apertus-Base ships with `rope_theta=12M, max_position=65536, llama3 scaling` (Path A). We trained with `rope_theta=500K, max_position=4096, no scaling` (Path B) inherited from the bakeoff. The cleanest claim is regime-vs-bakeoff (both on Path B). The Apertus-Base-vs-iter-1192 claim is geometry-mixed; matched-config eval is *perturbed*, not a clean baseline. Task-2 recommendation: switch to Path A — see `cpt-plan.md` §3.4 Q3.4.10 + `hyperparameters.json[training_geometry.task2_geometry_recommendation]`.
2. **Decontamination absent against the 4 native MCQ benchmark prompts.** Plan §6 V1 explicitly defers this for the diagnostic. Documented in 5 B report §10.2.
3. **Plutus QA −5.33 pp drop iter 834 → iter 1192.** Paired vs iter 834 CI [−0.1067, +0.0000] touches zero; paired vs Apertus-Base CI [−0.1422, −0.0133] outside zero. n=225. Diagnostic, ambiguous, leaning small-real.

## Compute

- **217.2 GPU-h total** (training 175.25 + sidecar evals 36.56 + matched-config base eval 2.74 + MCQ resubmits 2.01 + smokes 0.64; xfer watcher 0).
- 5 B tokens trained on 1 Clariden GH200 node (4 GPUs), TP=2, mb=2, global batch 4.194 M tokens.
- Real-world wall-clock from training start (2026-05-28 14:55 UTC) to iter 1192 done (~2026-05-30 12:00 UTC) ≈ 45 h.
- Throughput steady at ~8050 tokens/sec/GPU.

## What Task 2 should inherit

See `TASK2_HANDOFF.md` for the full handoff. Top items:

- Use **Path A geometry** (rope=12M, max_pos=65536, llama3 scaling). Removes the matched-config workaround. cpt-plan §3.4 Q3.4.10.
- Lock the **canonical eval task list** at the eval-submission layer, not at the reviewer-interpretation layer. Use `goal/canonical_eval_tasks.json` (or its Task-2 equivalent) as the authoritative spec.
- Wire **decontamination MinHash** against the 4 native MCQ prompts as a launch-blocker (V1).
- Rebuild the BPB heldout file to drop documents that overflow `max_context=4096` (~29 % currently truncated). Within-run trajectory deltas remain valid; absolute BPB is biased.
- Track **per-task** in addition to aggregate. greekmmlu / ilsp_medical_mcqa / ilsp_mcqa_asep / Plutus behave very differently; the aggregate hides task-level cancellation.
