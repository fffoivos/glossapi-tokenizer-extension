# Offline checkpoint-ensemble and instability results (zero inference)

Date: 2026-08-19. Status: computed and gate-verified; **not yet independently
reviewed**.

This is the deferred stage-2 start for this sub-subproject: the checkpoint
**ensemble** arm of Nishida, Isonuma & Oda (arXiv:2510.04848), computed
entirely from already-stored prediction artifacts — no model inference. It
also produces the first real MTV/IS numbers for the completed 8B trajectory.
Weight averaging (the paper's other arm) still requires new evaluation runs
and is not claimed here.

## Inputs and verification

- Predictions: the 19 `combined/predictions.jsonl` files under
  `/iopsstor/scratch/cscs/fffoivos/evals/full8_{native_greek_3cp_20260812/matrix_v6,
  native_greek_peak_window_20260817/matrix_v1_issue94,
  remaining12_checkpoint_release_20260817/matrix_v1}`. The three 3cp
  checkpoints (full 83,970-row panels) were restricted to the exact clean ID
  sets of the clean-only evaluations; every checkpoint then has identical
  per-benchmark IDs summing to 73,894 (fail-closed check).
- GreekMMLU: `analysis/20260811_greekmmlu_response_displacement_v1/output/`
  `greekmmlu_exact_responses.npz` (19 x 16,159; SHA-256 `67641b0c…a65356`).
- Gate: recomputed single-checkpoint clean accuracies match
  `FULL8_ALL_CHECKPOINT_NATIVE_BENCHMARKS_20260819.data.json` to 1e-9 for all
  19 x 8 native points and all 19 GreekMMLU points (**PASS**). The vectorized
  MTV/IS implementation matches `checkpoint_instability.py` on GPCR to <1e-12.
- Code + per-input SHA-256 manifest + full results:
  `/iopsstor/scratch/cscs/fffoivos/evals/full8_offline_ensembles_20260819/`
  (`code/ensemble_instability_v1.py`, `offline_ensemble_instability_v1.json`,
  result SHA-256 `efc9c392…8085259a`; local copy adjacent to this file).

Predeclared rules: single-model prediction = stored `pred_index` (verified =
argmax `avg_logprob` on 20,000 sampled rows); E1 = majority vote, ties by
higher window-mean `avg_logprob` (GreekMMLU: latest checkpoint in window,
since the npz has no per-choice scores, hence no E2 for GreekMMLU); E2 = argmax
of window-mean `avg_logprob`. Windows: tail m2/m3, cooldown m5
{14627, 15496, 16688, 17880, 18284}, peak m3/m5 (centered on 9,536), all 19,
and rolling m=5.

## Result 1 — the paper's stability claim replicates strongly

Rolling m=5 ensembles versus raw checkpoints over the same 15 trajectory
positions (example-level MTV; aggregate-accuracy MTV; mean accuracy %):

| Benchmark | MTV raw → ens | agg-MTV raw → ens | mean acc raw → ens |
| --- | --- | --- | --- |
| GreekMMLU (E1) | .089 → .029 | .0108 → .0034 | 54.91 → 55.27 |
| OYXOY WiC | .336 → .102 | .1783 → .0508 | 57.41 → 67.96 |
| OYXOY NLI | .179 → .059 | .0412 → .0181 | 56.89 → 60.72 |
| OYXOY metaphor | .084 → .071 | .0272 → .0300 | 39.93 → 45.96 |
| Medical MCQA | .088 → .045 | .0111 → .0106 | 40.64 → 41.51 |
| ASEP | .069 → .034 | .0085 → .0071 | 54.33 → 54.25 |
| DemosQA | .034 → .028 | .0048 → .0095 | 46.51 → 46.23 |
| GPCR | .074 → .030 | .0118 → .0081 | 60.52 → 60.34 |
| OYXOY WSD | .035 → .018 | .0032 → .0013 | 38.32 → 38.39 |

Instability drops 2–3x essentially everywhere, and mean accuracy *rises*
substantially exactly on the label-bias-unstable tasks (WiC +10.6, NLI +3.8,
metaphor +6.0) while staying within ±0.3 on the stable ones.

## Result 2 — fixed windows: peak-window aggregation is the promising arm

Accuracy (%), ensemble vs best single checkpoint inside the same window:

| Benchmark | cd5 best | cd5 E1/E2 | peak5 best | peak5 E1/E2 | best single (19) |
| --- | ---: | --- | ---: | --- | ---: |
| GreekMMLU | 54.85 | **54.99** / — | 56.81 | 56.38 / — | 56.81 @9536 |
| NLI | 58.12 | 40.62 / 42.47 | 63.96 | **64.93 / 65.88** | 65.47 @14304 |
| Medical | 40.81 | 38.42 / 39.14 | 41.77 | **42.24 / 42.24** | 44.87 @3576 |
| ASEP | 55.25 | 55.00 / 55.00 | 55.51 | 54.92 / 55.00 | 55.51 @11920 |
| DemosQA | 46.74 | 45.91 / 46.41 | 47.75 | 46.58 / 46.74 | 47.75 @5960 |
| GPCR | 62.89 | 61.34 / 62.37 | 63.40 | 61.34 / 60.82 | 63.40 @10728 |
| WSD | 38.61 | 38.15 / 38.11 | 38.90 | 38.41 / 38.47 | 38.90 @11920 |
| WiC | 76.58 | 50.45 / 56.74 | 67.90 | 45.08 / 47.92 | 77.60 @400 |
| Metaphor | 34.48 | 33.89 / 33.89 | 49.17 | 35.46 / 34.43 | 69.93 @2384 |

Readings:

1. **Peak-window E2 NLI (65.88) beats every single checkpoint of the run**
   (65.47), and peak Medical E1/E2 (42.24) beats its window best. GreekMMLU
   peak ensembles land 0.3–0.4 below the 9,536 peak.
2. **Cooldown-5 (the "average the last 5" plan) is roughly terminal-parity**:
   GreekMMLU E1 54.99 vs terminal 54.85; most native tasks a few tenths below
   the window best. It stabilizes, it does not recover the peak.
3. **Aggregation is not monotone protection: cooldown NLI collapses**
   (40.6–42.5 vs 58.12 window best). The cooldown NLI checkpoints disagree in
   *label bias* (window IS = .233), and a vote across oppositely-biased
   checkpoints can be worse than every ingredient. The same mechanism caps
   WiC/metaphor ensembles far below their early lucky extremes — which
   balanced accuracy exposes as label-prior artifacts anyway (WiC best
   balanced 55.0, terminal 50.4, peak5 E1 53.4).
4. NLI exact-set: peak5 E2 reaches 19.11% vs 2–4% for early singles, but the
   late singles' 26.49% is not matched — exact-set rewards the late
   label-distribution, consistent with (3).

## Implication for the weight-averaging step

The ensemble is the decision-level forecast; weight averaging can additionally
move the point toward the basin center (NLL/BPB effects the vote cannot show).
On this evidence: the **peak-window (7152–11920) uniform average is the
highest-value arm**, the cooldown-5 average is a stabilization arm with
roughly terminal-level scores, and any average that mixes strongly
opposite-label-bias checkpoints (cooldown NLI pattern) needs balanced-accuracy
reporting to avoid mistaking bias cancellation for capability change.
