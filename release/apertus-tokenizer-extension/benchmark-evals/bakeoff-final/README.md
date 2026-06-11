# Bakeoff final results — 5.0B endpoint

Generated UTC: `2026-05-26T16:00:00+00:00`. Canonical headline benchmark comparison across the 4 bakeoff arms at each arm's final iter.

For the iso-token Vanilla / ReTok / TD comparison at 3.5B, see `../3.5B-comparison/`.

Full narrative + per-task breakdown:
[`../../supporting-material/provenance/evals/BAKEOFF_FINAL_RESULTS_20260526.md`](../../supporting-material/provenance/evals/BAKEOFF_FINAL_RESULTS_20260526.md).

Reconciliation against the v0.12 experimental-design plan + 14-entry discrepancy log:
[`../../supporting-material/provenance/decisions/PLAN_VS_RESULTS_RECONCILIATION_20260526.md`](../../supporting-material/provenance/decisions/PLAN_VS_RESULTS_RECONCILIATION_20260526.md).

## Greek aggregate rule

Explicit MT diagnostics (`arc_challenge_mt_el`, `global_piqa_completions_ell_grek`) are **excluded** from the Greek aggregate per v0.12 §10 Q6 ("secondary, useful for Krikri-comparability only, not weighted heavily in our decision rule"). They remain visible in per-task tables in the full narrative doc. The 5-task no-MT Greek aggregate is computed from `global_mmlu_full_el`, `include_base_44_greek_few_shot_en`, `belebele_ell_Grek`, `xnli_el`, `xquad_el`.

## Loss-reading rule

Raw Megatron `lm loss` is per-token CE and is **not** tokenizer-fair across Vanilla's 131,072 vocab and the 148,480-vocab extended arms. Cross-arm conclusions use heldout BPB (bits per UTF-8 byte) plus downstream evals. Older artifacts may call BPB `BPC`; that is a legacy bits-per-byte label.

## Bottom line

- **TokenDistil-5B** is the bakeoff downstream winner: leads no-MT Greek aggregate (+1.28 pp), English retention (+1.04 pp), and Multilingual (+0.40 pp) over Vanilla-5B at iter 1192.
- **Vanilla-5B** retains tokenizer-fair heldout BPB leadership (0.4602 vs TD 0.4872; gap 0.027, narrowing).
- **Centroid** clearly broken at any scale (BPC stuck ~0.90; no-MT Greek 0.2566 vs ≥ 0.40 for others).
- **ReTok** dominated by TD on every shared iter; stopped at 3.5B.
- **TD's Greek lead is xquad_el-load-bearing** (+7.57 pp on that single task). Strip xquad_el and Vanilla wins the remaining 4 no-MT Greek tasks narrowly. Whether xquad_el is in the production-target set determines the headline.

## Final state per arm

| Arm | Iter | Tokens (B) | Greek no-MT | EN ret | Multi | BPB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla-5B    | 1192 | 5.000 | 0.4076 | 0.6799 | 0.4936 | **0.4602** |
| TokenDistil-5B | 1192 | 5.000 | **0.4204** | **0.6903** | **0.4976** | 0.4872 |
| ReTok-3.5B   | 834  | 3.498 | 0.3984 | 0.6786 | 0.4864 | 0.5390 |
| Centroid-2B  | 476  | 1.996 | 0.2566 | 0.6836 | 0.4888 | 0.8994 |

V4-HF Apertus base reference: Greek agg ≈ 0.525, EN ret ≈ 0.716, Multi ≈ 0.541. **All arms degrade vs base Apertus on downstream Greek aggregate**, even though all arms IMPROVE on BPB.

## 3.5B → 5.0B continuation deltas

| Arm | Greek no-MT d | EN retention d | Multilingual d | BPB d |
|---|---:|---:|---:|---:|
| Vanilla-5B    | +0.87 pp | +0.17 pp | +0.13 pp | -0.0122 |
| TokenDistil-5B | +0.75 pp | +0.38 pp | -0.09 pp | -0.0182 |

TD's BPB slope (−0.0182 / 1.5B) is 1.5× steeper than Vanilla's; linear-slope BPB crossover predicted at ~6.5B (possibly never under WSD cooldown attenuation).

## BPC trajectory across arms (heldout, lower better)

| Iter | Tokens (B) | Vanilla | ReTok | Centroid | TD |
|---:|---:|---:|---:|---:|---:|
| 130 | 0.545 | 0.5432 | 0.7561 | 1.1318 | 0.6531 |
| 476 | 1.996 | 0.4906 | 0.5739 | **0.8994** | 0.5311 |
| 834 | 3.498 | 0.4724 | **0.5390** | — | 0.5054 |
| 1192 | 5.000 | **0.4602** | — | — | **0.4872** |

TD-vs-Vanilla BPB gap: 0.110 (iter 130) → 0.033 (iter 834) → **0.027 (iter 1192)**. Monotonically narrowing.

## Important caveat: thresholds were not pre-committed

The v0.12 experimental-design plan (`old_experiments_plan.md` §10 Q8) committed to a pre-commit decision rule with 5 thresholds: X (preservation gate %), M_progress (Greek improvement floor %), M_ext (extension-beats-Vanilla %), M_van (Vanilla-beats-extension %), T (TD-beats-ReTok %). v0.12 §10 Q8f explicitly warned: *"Doing this with results visible risks post-hoc rationalization; the rule is only as honest as the pre-commitment."*

**None of the 5 thresholds were locked before results came in.** The 5B headline above is therefore an honest description of the numbers, not a v0.12-§10-Q8 adjudication.

For the full discrepancy log (14 entries, 6 HIGH-severity), see:
[`../../supporting-material/provenance/decisions/PLAN_VS_RESULTS_RECONCILIATION_20260526.md`](../../supporting-material/provenance/decisions/PLAN_VS_RESULTS_RECONCILIATION_20260526.md).

## Plots

See [`plots/`](plots/) for the regenerated 5B-endpoint visualizations: 4-arm group-averaged trajectories, per-task subplots, intrinsic-metric trajectories, and Vanilla-vs-TD focused comparisons.

## Artifacts on Clariden

Full eval JSONs + per-sample logs + intrinsic metrics + new-token diagnostics live remotely:

```text
/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_td_vs_vanilla_20260525T142522Z_*/
```

This release stores only compact summaries.
