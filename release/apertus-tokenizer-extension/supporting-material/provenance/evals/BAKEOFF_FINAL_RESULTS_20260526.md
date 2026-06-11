# 4-arm bakeoff — final consolidated results (iter 1192 / 5.0B for Vanilla + TD)

Generated UTC: `2026-05-26T06:30:00+00:00`. Supersedes:

- `BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md` (2B-only trajectory analysis)
- `CONTINUATION_3P5B_RESULTS_20260525.md` (3.5B continuation; still accurate at its tokenwise scope but no longer the headline)

**Post-native-suite update.** This report's Greek conclusion is based on the
older fallback lm-eval Greek slice, not the vetted native-Greek suite. The
native-Greek suite completed later on 2026-05-26 and changes the Greek-specific
selection reading: Vanilla is ahead of TD on the native MCQ headline, while TD
still leads the older fallback downstream/retention bundle. See the source repo
report `NATIVE_GREEK_SUITE_RESULTS_20260526.md`.

## What ran where

| Arm | Final iter | Final tokens (B) | Status |
|---|---:|---:|---|
| Vanilla | 1192 | 5.000 | Complete |
| TD layer11 | 1192 | 5.000 | Complete |
| ReTok | 834 | 3.498 | Stopped at 3.5B continuation |
| Centroid | 476 | 1.996 | Stopped at 2B bakeoff (clearly worst arm; not continued) |

Eval coverage at every iter in `[130, 260, 325*, 390, 455, 476, 585, 715, 834, 1013, 1192]` per arm where checkpoints existed. *Centroid + TD skipped iter 325. Vanilla/ReTok/TD have iters 585+; Centroid stops at 476. Only Vanilla and TD have iters 1013/1192.

Local artifact tree:

- `per_iter_results/{arm}_iter{it}.json` — lm-eval results.
- `per_iter_results/intrinsic/{arm}_iter{it:03d}_fair.json` — tokenizer-fair BPC / NLL.
- `per_iter_results/diagnostics/{retok,td}_iter{it}_new_token_diagnostics.json` — §5.3 new-token diagnostics.
- `per_iter_results/training_logs/*.out` — Megatron stdout for all 21 (bake + resume + 3.5B + 5B) jobs.

## Bottom line

Greek aggregate rule: explicit MT diagnostics (`arc_challenge_mt_el`,
`global_piqa_completions_ell_grek`) are excluded from aggregate calculations.
They remain visible in the per-task tables only.

**TD layer11 is the bakeoff winner on the older fallback downstream/retention
bundle.** At iter 1192:

- TD leads the no-explicit-MT Greek aggregate (+1.28 pp), English retention (+1.04 pp), and Multilingual (+0.40 pp) over Vanilla.
- TD's no-explicit-MT Greek lead emerged by iter 834 (+1.40 pp) and remains at iter 1192 (+1.28 pp). The 3.5B -> 5B continuation confirms the lead did not collapse, but it did not widen in this window.
- TD trails Vanilla only on tokenizer-fair BPC (0.4872 vs 0.4602; gap 0.0270), but the gap has narrowed every iter and TD's BPC slope is steeper than Vanilla's.

**Vanilla retains intrinsic-compression leadership.** Heldout BPC remains lowest for Vanilla, but the gap has shrunk from 0.110 (iter 130) → 0.033 (iter 834) → 0.027 (iter 1192). Linear-slope crossover is now extrapolated to ~6.5B tokens, possibly never if WSD cooldown fully attenuates both arms.

**ReTok and Centroid are ruled out.** Centroid is clearly broken (Greek 0.293 vs all others ≥0.42 from iter 130; BPC stuck at ~0.9 vs others ≤0.65). ReTok closes its BPC gap fastest among the extended arms but trails TD on downstream at every shared iter.

## Aggregate scoreboard

### Final state per arm (last iter each reached)

| Arm | Iter | Tokens (B) | Greek no-MT agg | EN retention | Multilingual | BPC ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla    | 1192 | 5.000 | 0.4076 | 0.6799 | 0.4936 | **0.4602** |
| TD layer11 | 1192 | 5.000 | **0.4204** | **0.6903** | **0.4976** | 0.4872 |
| ReTok      | 834  | 3.498 | 0.3984 | 0.6786 | 0.4864 | 0.5390 |
| Centroid   | 476  | 1.996 | 0.2566 | 0.6836 | 0.4888 | 0.8994 |

Bold = group winner. Comparing different arms across different token budgets is intentionally explicit; ReTok and Centroid were stopped early because their trajectories did not justify continued compute.

### Iso-token comparison at iter 834 (3.5B) — fair across Vanilla/ReTok/TD

| Arm | Greek no-MT agg | EN retention | Multilingual | BPC ↓ |
|---|---:|---:|---:|---:|
| Vanilla    | 0.3989 | 0.6782 | 0.4923 | **0.4724** |
| TD layer11 | **0.4129** | **0.6865** | **0.4967** | 0.5054 |
| ReTok      | 0.3984 | 0.6786 | 0.4864 | 0.5390 |

### Iso-token comparison at iter 476 (2.0B) — fair across all 4 arms

| Arm | Greek no-MT agg | EN retention | Multilingual | BPC ↓ |
|---|---:|---:|---:|---:|
| Vanilla    | **0.4131** | 0.6818 | **0.4901** | **0.4906** |
| TD layer11 | 0.4048 | 0.6828 | 0.4899 | 0.5311 |
| ReTok      | 0.3906 | 0.6750 | 0.4873 | 0.5739 |
| Centroid   | 0.2566 | **0.6836** | 0.4888 | 0.8994 |

At 2B (the original bakeoff budget) Vanilla was still the safe default. Extending to 3.5B/5B revealed the trajectory-derived TD advantage. This is the trajectory pivot referenced in `BAKEOFF_TRAJECTORY_ANALYSIS_20260524.md` §"Mid-window slope extrapolation".

## Continuation deltas (iter 834 → iter 1192, 3.5B → 5.0B)

| Arm | Greek no-MT d | EN retention d | Multilingual d | BPC d |
|---|---:|---:|---:|---:|
| Vanilla    | **+0.87 pp** | +0.17 pp | +0.13 pp | -0.0122 |
| TD layer11 | +0.75 pp | **+0.38 pp** | -0.09 pp | **-0.0182** |

TD outpaces Vanilla on EN retention and BPC slope; Vanilla slightly outpaces TD on the no-explicit-MT Greek aggregate during this final window, while TD keeps the larger absolute Greek lead. Both decay slightly on Multilingual in this window. TD's BPC slope (−0.0182 / 1.5B) is 1.5× steeper than Vanilla's (−0.0122 / 1.5B).

## Per-task winners at iter 1192

| Group | Task | Vanilla | TD layer11 | Winner | Gap |
|---|---|---:|---:|---|---:|
| EN ret | `mmlu` | 0.5362 | 0.5616 | **TD** | +2.54 pp |
| EN ret | `hellaswag` | 0.7574 | 0.7650 | **TD** | +0.76 pp |
| EN ret | `arc_easy` | 0.7862 | 0.7883 | **TD** | +0.21 pp |
| EN ret | `arc_challenge` | 0.5213 | 0.5367 | **TD** | +1.54 pp |
| EN ret | `piqa` | 0.8020 | 0.7911 | Vanilla | -1.09 pp |
| EN ret | `winogrande` | 0.6764 | 0.6993 | **TD** | +2.29 pp |
| Multi | `global_mmlu` | 0.4455 | 0.4604 | **TD** | +1.49 pp |
| Multi | `xcopa` | 0.6213 | 0.6202 | Vanilla | -0.11 pp |
| Multi | `xnli` | 0.4142 | 0.4123 | Vanilla | -0.19 pp |
| Greek | `global_mmlu_full_el` | 0.4214 | 0.4188 | Vanilla | -0.26 pp |
| Greek | `include_base_44_greek_few_shot_en` | 0.4384 | 0.4167 | Vanilla | -2.17 pp |
| Greek | `belebele_ell_Grek` | 0.5122 | 0.5378 | **TD** | +2.56 pp |
| Greek MT diag | `arc_challenge_mt_el` | 0.4275 | 0.4113 | Vanilla | -1.62 pp |
| Greek | `xnli_el` | 0.3884 | 0.3755 | Vanilla | -1.29 pp |
| Greek | `xquad_el` | 0.2775 | 0.3532 | **TD** | **+7.57 pp** |
| Greek MT diag | `global_piqa_completions_ell_grek` | 0.5900 | 0.5900 | Tie | 0.00 pp |

EN retention: TD wins 5/6, Vanilla 1/6 (piqa, surface knowledge).

Multilingual: TD 1/3, Vanilla 2/3 on a per-task basis — but TD wins the aggregate on the strength of global_mmlu.

Greek no-MT: per-task is Vanilla 3 / TD 2. **TD's Greek-aggregate win is driven primarily by xquad_el** (+7.57 pp, a reading-comprehension/extractive-QA task) plus Belebele (+2.56 pp). Without xquad_el, the remaining 4 no-MT Greek tasks put Vanilla narrowly ahead (0.4401 vs 0.4372). This is an important nuance — the headline "TD wins Greek" depends on whether xquad_el is treated as in-distribution for the production target.

## BPC trajectory (heldout, tokenizer-fair — lower is better)

| Iter | Tokens (B) | Vanilla | ReTok | Centroid | TD |
|---:|---:|---:|---:|---:|---:|
| 130 | 0.545 | 0.5432 | 0.7561 | 1.1318 | 0.6531 |
| 260 | 1.091 | 0.5173 | 0.6370 | 0.9875 | 0.5670 |
| 390 | 1.636 | 0.4958 | 0.5891 | 0.9280 | 0.5399 |
| 476 | 1.996 | **0.4906** | 0.5739 | **0.8994** | 0.5311 |
| 585 | 2.454 | 0.4838 | 0.5606 | — | 0.5208 |
| 715 | 2.999 | 0.4770 | 0.5474 | — | 0.5115 |
| 834 | 3.498 | 0.4724 | **0.5390** | — | 0.5054 |
| 1013 | 4.249 | 0.4657 | — | — | 0.4953 |
| 1192 | 5.000 | **0.4602** | — | — | **0.4872** |

TD-vs-Vanilla BPC gap: 0.110 → 0.050 → 0.044 → 0.041 → 0.037 → 0.034 → 0.033 → 0.030 → **0.027**. Gap is monotonically narrowing. Linear extrapolation from the 3.5B → 5B window predicts crossover at ~6.8B tokens.

## New-token diagnostics (TD layer11, heldout 500 docs, §5.3 D2/D5)

| Iter | Avg prob mass on new-target positions (D2) | Greedy new-token utilization (D5) |
|---:|---:|---:|
| 476  | 0.3425 | 20.8% |
| 585  | 0.3425 | 29.6% |
| 715  | 0.3421 | 25.6% |
| 834  | 0.3421 | 28.2% |
| 1013 | 0.3410 | 26.0% |
| 1192 | 0.3431 | 21.8% |

D2 (probability mass concentrated on new-token positions when a new token is the target): essentially flat at ~0.342 across the 2B → 5B window. The new-token rows hit their effective-mass plateau by iter 476 and don't continue to improve. This is consistent with TD's gains beyond 2B coming from base-vocab adaptation and contextual integration, not further new-token-row training.

D5 (greedy utilization rate, fraction of greedy-decoded tokens that are new-vocab): noisy in 0.21–0.30 range with no clean trend. Small sample (500 generations × 5 prompts), so this metric is high-variance.

ReTok diagnostics for iter 585/715/834 are in `per_iter_results/diagnostics/retok_iter*_new_token_diagnostics.json` and are summarized in `CONTINUATION_3P5B_RESULTS_20260525.md`. ReTok's D2 trails TD's by ~3 pp at every shared iter; no ReTok diagnostics at iter 1013/1192 because ReTok stopped at 3.5B.

## What changed from the 3.5B picture

Two things to flag explicitly:

1. **TD's no-explicit-MT Greek lead is durable, not a one-checkpoint artifact.** At iter 834 it was TD +1.40 pp. At iter 1192 it remains TD +1.28 pp, driven by xquad_el +7.57 pp and Belebele +2.56 pp while INCLUDE-44 and XNLI move against TD. The 5B continuation confirms TD's lead survived more training, though it did not widen after 3.5B.
2. **TD's EN-retention lead widened.** At iter 834: TD +0.83 pp. At iter 1192: TD +1.04 pp. The "extended-vocab arms damage English knowledge" worry is contradicted by the data: TD actively *helps* English retention relative to Vanilla through the 5B window.

What did **not** change:

- Vanilla still wins tokenizer-fair BPC.
- Vanilla still wins surface-knowledge-style benchmarks (piqa, xnli, arc_challenge_mt_el).
- TD's per-task Greek wins remain narrow (2/7 explicit, plus 1 tie, with xquad_el as the swing).

## Production-decision implications

Per `PRODUCTION_DECISION_STATE.md` (currently states "Vanilla is the safe default at 2B"):

- **At 5B, Vanilla is no longer the unambiguous safe default.** Vanilla wins BPC and 3 of 5 no-MT Greek per-task; TD wins aggregate no-MT Greek + EN + Multi and 5 of 6 EN per-task. Choice depends on whether the production criterion weighs aggregate downstream over per-task or vice versa.
- **TD has not regressed on any aggregate.** This makes the cost-benefit argument for TD strong: the extra init cost is fixed (one-time per checkpoint), the inference cost is identical (148,480-vocab softmax vs 131,072-vocab is sub-1% throughput delta), and the downstream upside is real.
- **The 5B point validates the bakeoff design.** The 2B budget was sufficient to RANK the arms (TD/Vanilla close, ReTok/Centroid clearly behind) but NOT to PICK between TD and Vanilla. Extending to 3.5B was load-bearing for the no-MT Greek aggregate reversal; extending to 5B confirmed the TD lead persisted.

## Artifact checklist

Local (this directory, all eval JSONs present):

- `per_iter_results/{vanilla,td}_iter{1013,1192}.json` — lm-eval at 5B-continuation iters
- `per_iter_results/intrinsic/{vanilla,td}_iter{1013,1192}_fair.json` — BPC/NLL
- `per_iter_results/diagnostics/td_iter{1013,1192}_new_token_diagnostics.json` — §5.3 D2/D5
- `per_iter_results/training_logs/5b_{vanilla,td_layer11}_{1013,1192}-*.out` — Megatron stdout

Plots regenerated (this directory, `plots/`):

- `trajectories.png` — 4-arm group-averaged trajectories with 5B coverage
- `trajectories_per_task.png` — 8 individual benchmark panels
- `loss_comparison_van_td.png` — fair-vs-unfair loss comparison (3-panel)
- `loss_comparison_4arm.png` — 4-arm version of the above
- `training_loss.png`, `training_loss_logy.png`, `training_loss_van_td.png` — raw training-loss traces
- `intrinsic_trajectories.png`, `intrinsic_van_td.png` — tokenizer-fair intrinsic metrics
- `global_mmlu_full_el_subcategories_van_td.png`, `include_base_44_greek_subjects_van_td.png` — per-subject breakdowns

Remote (Clariden, full eval bundles):

- `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_5b_*` — packed eval outputs (full lm-eval samples + perplexity dumps), still available for re-evaluation if needed

## Pending items

- **Per-task confidence intervals.** lm-eval emits `*_stderr,none` fields; none of the aggregates above carry them. For the production-decision write-up these should be propagated through, especially around the xquad_el-dependent Greek headline.
- **Truncation-bias re-evaluation of BPC.** Vanilla truncates 29.2% of heldout docs at 4096-token context; TD truncates 24.8%. Vanilla's BPC is computed on a slightly easier subset (4.18M bytes vs TD's 5.46M bytes). A matched-byte-budget re-evaluation would tell us how much of the 0.027 gap is real vs methodological.
- **ReTok continuation to 5B (optional).** ReTok was stopped at 3.5B because TD dominated it; if the production decision turns out to need ReTok as a backup, a 1.5B continuation is cheap and would resolve "does ReTok's BPC slope catch TD eventually."
- **PF5** (separately tracked): merge ILSP Greek YAMLs (`hellaswag_greek`, `winogrande_greek`, `mmlu_pro_greek`, `truthfulqa_greek`, `medical_mcqa_greek`) into the lm-eval-harness clone so future evals have native-Greek MCQ coverage beyond INCLUDE-44.
