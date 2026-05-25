# 3.5B continuation results - Vanilla vs ReTok vs TD layer11

Generated UTC: `2026-05-25T05:23:53+00:00`.

This summarizes the continuation run `continuation_3p5b_20260524T143012Z`,
which extended Vanilla, ReTok, and TD layer11 from iter 476 (~2.0B tokens)
to iter 834 (~3.5B tokens). Local JSON snapshots live under
`per_iter_results/`; remote full artifacts remain on Clariden under
`/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_*`.

Loss-reading rule: raw Megatron `lm loss` is per-token CE and is not
tokenizer-fair across Vanilla vs the 148,480-vocab arms. This report therefore
uses heldout BPC/BPB and downstream evals for cross-arm conclusions; raw
training loss plots are diagnostic-only.

## Bottom line

- TD layer11 is the best final benchmark arm overall: it is first on English
  retention and multilingual aggregates, and narrowly first on the Greek
  aggregate at iter 834.
- Vanilla still has the best tokenizer-fair heldout Greek BPC, but its
  downstream Greek aggregate declined during the 2.0B -> 3.5B continuation.
- ReTok improves fastest on BPC and wins Greek MMLU / INCLUDE-44 Greek at
  iter 834, but it remains behind TD and Vanilla on the Greek aggregate.
- If selecting for the actual downstream bakeoff objective, TD layer11 is now
  the leading candidate. If selecting only for heldout BPC, Vanilla remains
  ahead.

## Aggregate scoreboard at iter 834

| Arm | Greek agg | Delta vs 476 | EN retention | Delta vs 476 | Multilingual | Delta vs 476 | BPC lower better | BPC delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 0.4339 | -0.70 pp | 0.6782 | -0.36 pp | 0.4923 | +0.22 pp | 0.4724 | -0.0182 |
| ReTok | 0.4246 | +0.96 pp | 0.6786 | +0.36 pp | 0.4864 | -0.09 pp | 0.5390 | -0.0349 |
| TD layer11 | 0.4344 | +0.90 pp | 0.6865 | +0.38 pp | 0.4967 | +0.68 pp | 0.5054 | -0.0256 |

## Per-task winners at iter 834

| Group | Task | Vanilla | ReTok | TD layer11 | Winner |
|---|---|---:|---:|---:|---|
| EN retention | `mmlu` | 0.5330 | 0.5565 | 0.5556 | ReTok |
| EN retention | `hellaswag` | 0.7573 | 0.7556 | 0.7632 | TD layer11 |
| EN retention | `arc_easy` | 0.7866 | 0.7576 | 0.7837 | Vanilla |
| EN retention | `arc_challenge` | 0.5230 | 0.5247 | 0.5358 | TD layer11 |
| EN retention | `piqa` | 0.7927 | 0.7851 | 0.7894 | Vanilla |
| EN retention | `winogrande` | 0.6764 | 0.6922 | 0.6914 | ReTok |
| Multilingual | `global_mmlu` | 0.4505 | 0.4603 | 0.4628 | TD layer11 |
| Multilingual | `xcopa` | 0.6178 | 0.5920 | 0.6156 | Vanilla |
| Multilingual | `xnli` | 0.4087 | 0.4070 | 0.4117 | TD layer11 |
| Greek | `global_mmlu_full_el` | 0.4145 | 0.4153 | 0.4036 | ReTok |
| Greek | `include_base_44_greek_few_shot_en` | 0.4022 | 0.4058 | 0.4004 | ReTok |
| Greek | `belebele_ell_Grek` | 0.5078 | 0.4933 | 0.5389 | TD layer11 |
| Greek | `arc_challenge_mt_el` | 0.4130 | 0.3899 | 0.4061 | Vanilla |
| Greek | `xnli_el` | 0.3831 | 0.3695 | 0.3851 | TD layer11 |
| Greek | `xquad_el` | 0.2868 | 0.3084 | 0.3364 | TD layer11 |
| Greek | `global_piqa_completions_ell_grek` | 0.6300 | 0.5900 | 0.5700 | Vanilla |

## Change from iter 476 to iter 834

| Group | Task | Vanilla delta | ReTok delta | TD layer11 delta |
|---|---|---:|---:|---:|
| EN retention | `mmlu` | -0.09 pp | +0.24 pp | +0.56 pp |
| EN retention | `hellaswag` | -0.21 pp | +0.69 pp | +0.27 pp |
| EN retention | `arc_easy` | +0.13 pp | -0.59 pp | +0.29 pp |
| EN retention | `arc_challenge` | -0.43 pp | +0.94 pp | +0.85 pp |
| EN retention | `piqa` | -0.22 pp | -0.11 pp | -0.27 pp |
| EN retention | `winogrande` | -1.34 pp | +1.03 pp | +0.55 pp |
| Multilingual | `global_mmlu` | -0.64 pp | +0.14 pp | +1.34 pp |
| Multilingual | `xcopa` | +1.09 pp | -0.71 pp | +0.42 pp |
| Multilingual | `xnli` | +0.22 pp | +0.30 pp | +0.28 pp |
| Greek | `global_mmlu_full_el` | -0.69 pp | +1.62 pp | +1.77 pp |
| Greek | `include_base_44_greek_few_shot_en` | -1.63 pp | +3.26 pp | -0.36 pp |
| Greek | `belebele_ell_Grek` | -0.56 pp | -0.33 pp | +1.11 pp |
| Greek | `arc_challenge_mt_el` | -0.77 pp | +1.79 pp | +0.26 pp |
| Greek | `xnli_el` | -1.89 pp | -0.56 pp | +0.48 pp |
| Greek | `xquad_el` | -2.33 pp | -0.08 pp | +1.02 pp |
| Greek | `global_piqa_completions_ell_grek` | +3.00 pp | +1.00 pp | +2.00 pp |

## New-token diagnostics

Measured on the 500-document heldout slice. Top-k is the fraction of
positions whose correct new token appears in the model's top-k predictions;
lower mean rank is better.

| Arm | Top-1 at new target | Delta from 585 | Top-5 at new target | Delta from 585 | Mean rank | Delta from 585 |
|---|---:|---:|---:|---:|---:|---:|
| ReTok | 0.3799 | +2.00 pp | 0.5469 | +2.11 pp | 228.9 | -31.4 |
| TD layer11 | 0.4105 | +1.43 pp | 0.5811 | +1.54 pp | 174.3 | -18.8 |

## Artifact checklist

- Local packed-eval snapshots: `per_iter_results/{vanilla,retok,td}_iter{585,715,834}.json`.
- Local BPC snapshots: `per_iter_results/intrinsic/*_iter{585,715,834}_fair.json`.
- Local new-token diagnostics: `per_iter_results/diagnostics/{retok,td}_iter{585,715,834}_new_token_diagnostics.json`.
- Regenerated plots are written to `plots/`.
- Final remote packed eval job: `2376082`, state `COMPLETED`, exit `0:0`, elapsed `00:59:51`.
