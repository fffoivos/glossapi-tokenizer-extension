# 09.1 — Downstream-task instability, ensembles and checkpoint averaging

> **In one line:** asked whether the update-9,536 GreekMMLU peak was a capability or an artefact of checkpoint noise; found that most checkpoint-to-checkpoint movement *is* noise, and that a uniform weight average of the five peak-window checkpoints matches the peak while forgetting substantially less.
> **Period:** 2026-08-17 → 2026-08-20. **Status:** completed; **never committed by its author and never independently reviewed** — recovered from an uncommitted working tree on 2026-09-01 (`2aec4a66`), and both results documents carry that caveat in their own status lines.
> **Came from / led to:** the mid-run peak in [`../RESULTS.md`](../RESULTS.md) → this → the checkpoint-average branches published on the private Hugging Face repo staged by [`../09_2_checkpoint_trajectory_release/`](../09_2_checkpoint_trajectory_release/).

## Why this existed

The three-checkpoint screen said GreekMMLU peaks at update 9,536 and declines. Nishida, Isonuma & Oda, *Instability in Downstream Task Performance During LLM Pretraining* ([arXiv:2510.04848](https://arxiv.org/abs/2510.04848)), argues that such trajectories are dominated by per-example instability rather than by capability, and that checkpoint aggregation both measures and removes it. This sub-subproject applied that framework to the run's own saved artifacts.

## History

### 2026-08-17 — metrics, then the peak window

`checkpoint_instability.py` implements the paper's expressions (1) mean total variation and (2) instability score, with the mappings this run's artifacts allow: `L(θ)` is the stored per-example `correct` value, model output is the stored `pred_index`, and `sim` is exact match — so IS is the fraction of adjacent transitions at which the selected answer changes. Example-level MTV is the reported quantity; aggregate-accuracy MTV is kept as a separate diagnostic precisely because simultaneous gains and losses cancel in it. `analyze_predictions.py` is the adapter over `predictions.jsonl` and fails closed on duplicate IDs, missing examples, non-finite scores or fewer than two checkpoints. The paper's "last 20 % of checkpoints" window selection was deliberately kept *outside* the formulas, to be supplied and recorded by the caller.

The same day, the four saved checkpoints bracketing the peak (7,152 / 8,344 / 10,728 / 11,920 — about 30, 35, 45 and 50 B token slots) were scored on the clean subset; update 9,536 was reused, not rerun. See [`evaluation/README.md`](evaluation/README.md) and [`evaluation/PEAK_WINDOW_CLEAN_RESULTS_20260817.md`](evaluation/PEAK_WINDOW_CLEAN_RESULTS_20260817.md). The verdict: **there is no universal best checkpoint** — ASEP and GPCR improve later, OYXOY metaphor regresses almost monotonically after 30 B, and OYXOY WiC peaks sharply at 45 B and collapses at 50 B while its *balanced* accuracy barely moves, which is a label-bias signature rather than a capability trajectory.

### 2026-08-19 — the ensemble arm, computed from stored predictions only

[`evaluation/OFFLINE_ENSEMBLE_RESULTS_20260819.md`](evaluation/OFFLINE_ENSEMBLE_RESULTS_20260819.md) ran the paper's ensemble arm with **zero model inference**, over the 19 stored prediction files restricted to identical per-benchmark IDs summing to 73,894. The gate: recomputed single-checkpoint accuracies matched the published 19-checkpoint payload to 1e-9 on all 19 × 8 native points and all 19 GreekMMLU points, and the vectorized MTV/IS matched `checkpoint_instability.py` to <1e-12.

Findings: rolling five-checkpoint ensembles cut example-level MTV **2–3× on every benchmark** (e.g. WiC .336 → .102, GreekMMLU .089 → .029), and mean accuracy *rose* on exactly the label-bias-unstable tasks (WiC +10.6, metaphor +6.0, NLI +3.8 pp) while staying within ±0.3 pp on the stable ones. On fixed windows, peak-window E2 on OYXOY NLI (65.88 %) beat every single checkpoint of the run, while the cooldown-window ensemble **collapsed** on NLI (40.6–42.5 vs a window best of 58.12) because its checkpoints disagree in label bias and a vote across oppositely-biased members can be worse than any ingredient. The document's own implication was recorded before the experiment that tested it: the peak-window uniform average is the highest-value arm.

### 2026-08-19 → 08-20 — weight averaging, and it worked

[`CHECKPOINT_AVERAGE_RESULTS_20260819.md`](CHECKPOINT_AVERAGE_RESULTS_20260819.md) built two uniform means (fp32 accumulation, bf16 output, config/tokenizer identity verified): `avg_peak5` over iters 7,152–11,920 and `avg_cooldown5` over the five cooldown checkpoints, then scored both on the frozen native suite, on clean GreekMMLU and on the retention suite.

- `avg_peak5` **ties the run's GreekMMLU peak** — 56.78 % vs 56.81 % at n=16,159 (stderr ≈ 0.39 pp) — sets an all-time ASEP best (56.19 % vs 55.51 %), beats every in-window single on Medical and NLI, and beats the 40 B single on **all nine** retention tasks (macro 64.58 vs 63.68; terminal 62.95), recovering roughly half the forgetting relative to the early-checkpoint optimum.
- `avg_cooldown5` did what the ensemble forecast said: stabilization at roughly terminal level, no retention recovery, and a real NLI regression (43.34 %) reproducing the ensemble's label-bias collapse in weight space.
- A **correction dated 2026-08-20** is recorded in the same file: the terminal column had borrowed 50 B values; against the true iter-18,284 numbers `avg_cooldown5` beats terminal broadly (WiC +28.0, NLI +4.6, Medical +0.96).
- Also on 2026-08-20, both averages were uploaded as private branches `18-avg-uniform5-tokens30B-50B` and `19-avg-uniform5-tokens61B-77B`, the root `CHECKPOINTS.md` and `checkpoint-index.json` gained an averages table, and `default_revision` moved from `08-step9536-tokens40B` to the peak-window average. Everything stayed private.

## Outcome

- Instability is the dominant component of the run's benchmark trajectory on the label-sensitive tasks, and aggregation removes two thirds of it. Balanced accuracy must be reported alongside raw accuracy for OYXOY WiC, metaphor and NLI or bias cancellation is mistaken for capability change.
- The best single artifact the run produced is not a checkpoint: it is `avg_peak5`, which is at or above the best in-window single on 3 of 8 native tasks (one all-time best), near-parity elsewhere, and strictly better on retention than any checkpoint from 20 B onward. It became the repository default revision.
- Left open at the end: NLI exact-set and balanced-accuracy columns from the merged predictions; adoption of these numbers into the canonical presentations (they are **not** in the 19-checkpoint report); optional greedy-soup and peak-average-to-terminal interpolation arms; and independent review of the whole two-day sequence.

## Where things are

| What | Path |
| --- | --- |
| The two formulas, validated | [`checkpoint_instability.py`](checkpoint_instability.py), tests in [`test_checkpoint_instability.py`](test_checkpoint_instability.py) |
| Adapter over stored `predictions.jsonl` | [`analyze_predictions.py`](analyze_predictions.py) (`--ids` restricts every checkpoint to one clean set) |
| Peak-window evaluation | [`evaluation/`](evaluation/) — see its own README |
| Ensemble + first MTV/IS numbers | [`evaluation/OFFLINE_ENSEMBLE_RESULTS_20260819.md`](evaluation/OFFLINE_ENSEMBLE_RESULTS_20260819.md), payload [`evaluation/offline_ensemble_instability_v1.json`](evaluation/offline_ensemble_instability_v1.json) |
| Weight-average results and publication record | [`CHECKPOINT_AVERAGE_RESULTS_20260819.md`](CHECKPOINT_AVERAGE_RESULTS_20260819.md) |
| Peak-window report | [`presentations/NATIVE_GREEK_PEAK_WINDOW_5CP_20260817.html`](presentations/NATIVE_GREEK_PEAK_WINDOW_5CP_20260817.html) |
| The retention numbers these arms are judged against | [`../evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md`](../evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md) |

The merged model weights, the merge code and the per-shard evaluation receipts stayed on CSCS at the paths named inside `CHECKPOINT_AVERAGE_RESULTS_20260819.md`.
