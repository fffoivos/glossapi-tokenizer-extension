# 5B continuation results - Vanilla vs TD layer11

Generated UTC: `2026-05-26T12:19:57+00:00`.

Final 1192 artifacts are present.

**Post-native-suite update.** The planned native-Greek suite has now completed;
this file's statement that it was not run is historical to the original 5B
continuation report. For Greek-specific selection, use the source repo report
`NATIVE_GREEK_SUITE_RESULTS_20260526.md`: Vanilla leads TD on the vetted native
MCQ headline, while TD remains better on the older fallback downstream and
retention aggregates reported here.

Run tag: `continuation_5b_td_vs_vanilla_20260525T142522Z`.

Loss-reading rule: raw Megatron `lm loss` is per-token CE and is not
tokenizer-fair across Vanilla vs the 148,480-vocab TD arm. This report
uses heldout BPB and downstream evals for cross-arm conclusions.

Historical evaluation-scope warning for this report generation: the planned
native-Greek suite had not yet run when this file was first produced. The Greek
aggregate below is therefore the older fallback slice. Use the source repo
report `NATIVE_GREEK_SUITE_RESULTS_20260526.md` for the current native-Greek
headline.

## Available Checkpoints

| Iter | Tokens B | Vanilla eval | TD eval | Vanilla BPB | TD BPB | BPB winner |
|---:|---:|---:|---:|---:|---:|---|
| 476 | 1.996 | yes | yes | 0.4906 | 0.5311 | Vanilla |
| 834 | 3.498 | yes | yes | 0.4724 | 0.5054 | Vanilla |
| 1013 | 4.249 | yes | yes | 0.4657 | 0.4953 | Vanilla |
| 1192 | 5.000 | yes | yes | 0.4602 | 0.4872 | Vanilla |

## Aggregate Trajectory

| Iter | Greek no-MT Vanilla | Greek no-MT TD | Delta TD-V | EN Vanilla | EN TD | Delta TD-V | Multi Vanilla | Multi TD | Delta TD-V |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 476 | 0.4131 | 0.4048 | -0.0082 | 0.6818 | 0.6828 | +0.0010 | 0.4901 | 0.4899 | -0.0002 |
| 834 | 0.3989 | 0.4129 | +0.0140 | 0.6782 | 0.6865 | +0.0083 | 0.4923 | 0.4967 | +0.0044 |
| 1013 | 0.4056 | 0.4193 | +0.0138 | 0.6791 | 0.6892 | +0.0101 | 0.4904 | 0.4974 | +0.0069 |
| 1192 | 0.4076 | 0.4204 | +0.0128 | 0.6799 | 0.6903 | +0.0104 | 0.4936 | 0.4976 | +0.0040 |

## Per-Task Matched Comparison At Iter 1192

| Group | Task | Vanilla | TD | Delta TD-V | Winner | TD vs V4-HF | Vanilla vs V4-HF |
|---|---|---:|---:|---:|---|---:|---:|
| EN retention | MMLU | 0.5362 | 0.5616 | +0.0254 | TD | -0.0307 | -0.0561 |
| EN retention | HellaSwag | 0.7574 | 0.7650 | +0.0076 | TD | -0.0234 | -0.0310 |
| EN retention | ARC Easy | 0.7862 | 0.7883 | +0.0021 | TD | -0.0480 | -0.0501 |
| EN retention | ARC Challenge | 0.5213 | 0.5367 | +0.0154 | TD | -0.0503 | -0.0657 |
| EN retention | PIQA | 0.8020 | 0.7911 | -0.0109 | Vanilla | -0.0081 | +0.0028 |
| EN retention | Winogrande | 0.6764 | 0.6993 | +0.0229 | TD | +0.0063 | -0.0166 |
| Multilingual | Global MMLU | 0.4455 | 0.4604 | +0.0149 | TD | -0.0642 | -0.0791 |
| Multilingual | XCOPA | 0.6213 | 0.6202 | -0.0011 | Vanilla | -0.0373 | -0.0362 |
| Multilingual | XNLI | 0.4142 | 0.4123 | -0.0019 | Vanilla | -0.0277 | -0.0258 |
| Greek | Greek MMLU | 0.4214 | 0.4188 | -0.0026 | Vanilla | -0.0967 | -0.0941 |
| Greek | INCLUDE-44 Greek | 0.4384 | 0.4167 | -0.0217 | Vanilla | -0.0887 | -0.0670 |
| Greek | Belebele Greek | 0.5122 | 0.5378 | +0.0256 | TD | -0.0989 | -0.1245 |
| Greek | ARC Challenge MT-el | 0.4275 | 0.4113 | -0.0162 | Vanilla | -0.0682 | -0.0520 |
| Greek | XNLI Greek | 0.3884 | 0.3755 | -0.0129 | Vanilla | -0.0229 | -0.0100 |
| Greek | XQuAD Greek F1 | 0.2775 | 0.3532 | +0.0758 | TD | -0.1640 | -0.2397 |
| Greek | PIQA Greek MT | 0.5900 | 0.5900 | +0.0000 | tie | -0.0300 | -0.0300 |

## Greek Aggregate Variants At Iter 1192

At the time this report was generated, the planned native-Greek benchmark suite
had not run. The table below separates the all-available SwissAI fallback
bundle from the no-explicit-MT fallback slice. This is now historical evidence,
not the current native-Greek headline.

| Variant | Vanilla | TD | Delta TD-V | Note |
|---|---:|---:|---:|---|
| SwissAI 7-task fallback bundle | 0.4365 | 0.4433 | +0.0069 | Diagnostic only; includes two explicitly machine-translated tasks and is not the planned native-Greek suite. |
| Headline no-explicit-MT Greek slice | 0.4076 | 0.4204 | +0.0128 | Drops `arc_challenge_mt_el` and `global_piqa_completions_ell_grek`; still does not include greek-nlp/benchmark, Medical MCQA Greek, or OYXOY. |
| No-MT/no-XNLI diagnostic slice | 0.4124 | 0.4316 | +0.0193 | Also drops XNLI Greek because it is translated NLI; use as a sensitivity check only. |

## TD New-Token Diagnostics

| Iter | Top1 new target | Top5 new target | Mean rank | New-vocab mass | Greedy new-token use |
|---:|---:|---:|---:|---:|---:|
| 476 | 0.3864 | 0.5557 | 206.4 | 0.3425 | 0.2080 |
| 834 | 0.4105 | 0.5811 | 174.3 | 0.3421 | 0.2820 |
| 1013 | 0.4196 | 0.5903 | 162.7 | 0.3410 | 0.2600 |
| 1192 | 0.4278 | 0.5993 | 153.2 | 0.3431 | 0.2180 |

## Baseline Anchors

Baseline values come from `../V4_BENCHMARK_COMPARISON.md`.

| Baseline | Greek no-MT agg | EN retention | Multilingual |
|---|---:|---:|---:|
| V4-HF | 0.5146 | 0.7160 | 0.5407 |
| V4-postconv | 0.1978 | 0.3420 | 0.3629 |

## Decision Status

- Recommendation: Primary decision: TD has not overtaken the initial Vanilla/V4-HF scores by 5B. Secondary result: TD is the matched-5B downstream winner over continued Vanilla on the available fallback eval slices, including the no-explicit-MT Greek slice, while Vanilla remains better on heldout byte-normalized loss.
- Objective answer: No: TD does not overtake initial Vanilla/V4-HF by 5B. It overtakes matched continued Vanilla on downstream aggregates, including the available no-explicit-MT fallback Greek slice, but remains below V4-HF on the no-explicit-MT fallback Greek, English-retention, and multilingual aggregates. The later native-Greek suite favors Vanilla over TD on the native MCQ headline.
- Matched final aggregate deltas TD - Vanilla: Greek no-MT `+0.0128`, EN `+0.0104`, Multilingual `+0.0040`, BPB `+0.0270` (lower BPB is better).
- Per-task wins at iter 1192: TD `8`, Vanilla `7`, ties `1`.
- TD change since 3.5B: Greek `+0.0075`, EN `+0.0038`, Multilingual `+0.0009`.
- TD vs V4-HF baseline at final: Greek no-MT `-0.0942`, EN `-0.0257`, Multilingual `-0.0431`.
- Recovery reading: TD is still below original V4-HF on all three fallback aggregates, so it has not beaten the initial Vanilla/original Apertus scores by 5B. The 3.5B -> 5B trajectory is positive for TD on fallback downstream aggregates, especially Greek, but BPB still favors Vanilla. The later native-Greek suite should be used for the final native-benchmark claim.

## Linear Gap-Closure Sense Check

This is a rough extrapolation from only the 3.5B -> 5B interval,
not a forecast. It answers whether the observed slope is remotely
fast enough to catch V4-HF.

| Group | TD gap to V4-HF at 5B | TD gain 3.5B->5B | Extra B tokens at same slope | Total B tokens at same slope |
|---|---:|---:|---:|---:|
| Greek | 0.0942 | 0.0075 | 18.8 | 23.8 |
| EN retention | 0.0257 | 0.0038 | 10.2 | 15.2 |
| Multilingual | 0.0431 | 0.0009 | 71.6 | 76.6 |
