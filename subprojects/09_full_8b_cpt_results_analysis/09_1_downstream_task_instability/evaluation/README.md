# 09.1 evaluation/ — the peak-window native-Greek matrix

> **In one line:** scored the four saved checkpoints bracketing the GreekMMLU optimum so the "peak" could be seen as a window rather than a point; it showed there is no universal best checkpoint.
> **Period:** 2026-08-17 (with the offline ensemble work of 2026-08-19 landing here too). **Status:** completed and receipt-verified; not independently reviewed.

## What was evaluated

Four missing exports around the optimum, with update 9,536 reused rather than rerun:

| Position | Update | Token slots |
| --- | ---: | ---: |
| best − 2 | 7,152 | 29,997,662,208 |
| best − 1 | 8,344 | 34,997,272,576 |
| best | 9,536 | 39,996,882,944 (reused) |
| best + 1 | 10,728 | 44,996,493,312 |
| best + 2 | 11,920 | 49,996,103,680 |

Benchmark, prompts and scorer are unchanged from the frozen three-checkpoint screen in [`../../evaluation/`](../../evaluation/) — DemosQA, Medical MCQA, ASEP MCQA, GPCR and the four OYXOY views on the FP32 legacy candidate scorer. Because these benchmarks were *not* removed from the training corpus, only the clean subset is scored: the 10,076 strong-match exclusions from the original audit are rebound into a 73,894-row question file before the model loads, and full-panel scores are neither computed nor reported. Protipa stayed excluded behind its access gate.

## History

The plan was six sequential four-node debug segments of fourteen shards each. Live execution showed a 22-minute four-node envelope was insufficient for two long shards; completed receipts were preserved and only the missing shards and remaining segments were rerun on a receipt-verified two-node profile with longer wall time, inside the same node-minute ceiling. A separate post-processing defect counted four `combined/receipt.json` files as shard receipts; the filter was corrected and finalization was rerun with **zero model inference**. Neither correction changed examples, scorer settings or predictions. The final audit shows 84 unique shard receipts, 5 checkpoint rows and 9 reported benchmark views, with every referenced receipt hash reverified.

On 2026-08-19 the offline ensemble stage was computed from stored predictions only and its results and payload were written here as well ([`OFFLINE_ENSEMBLE_RESULTS_20260819.md`](OFFLINE_ENSEMBLE_RESULTS_20260819.md), [`offline_ensemble_instability_v1.json`](offline_ensemble_instability_v1.json)).

## Findings

[`PEAK_WINDOW_CLEAN_RESULTS_20260817.md`](PEAK_WINDOW_CLEAN_RESULTS_20260817.md) records the full accuracy and choice-NLL tables. The readings: the 40 B GreekMMLU-selected checkpoint is not the optimum for most of these tasks; ASEP and GPCR improve later (GPCR accuracy peaks at 45 B, NLL still improving at 50 B); OYXOY metaphor regresses almost monotonically after 30 B; OYXOY WiC peaks sharply at 45 B (67.90 %) and collapses at 50 B (33.77 %) while balanced accuracy barely moves; OYXOY NLI exact-set rises from 2.23 % to 26.49 % while raw item accuracy falls late. DemosQA and Medical accuracy are comparatively flat, though Medical choice NLL improves clearly by 50 B.

## Files

| File | Role |
| --- | --- |
| [`peak_window_checkpoint_bindings.json`](peak_window_checkpoint_bindings.json) | the four exports and their conversion receipts |
| [`bind_peak_window_native_suite.py`](bind_peak_window_native_suite.py) | validates the exports, rebinds the frozen examples without changing benchmark or scoring fields |
| [`run_peak_window_segment.sbatch`](run_peak_window_segment.sbatch), [`freeze_and_preflight_peak_window.sbatch`](freeze_and_preflight_peak_window.sbatch), [`smoke_peak_window.sbatch`](smoke_peak_window.sbatch) | the receipt-bound segmented evaluator and its preflight |
| [`coordinate_peak_window_segments.sh`](coordinate_peak_window_segments.sh) | bounded Mac-side submission loop for Clariden's two-submitted-debug-job limit |
| [`finalize_peak_window.py`](finalize_peak_window.py) | joins the four new results with the existing update-9,536 result |
| [`test_peak_window_evaluation.py`](test_peak_window_evaluation.py) | regression tests |

The runtime scorer itself lives in the already qualified immutable bundle; these files are an experiment-owned adapter for a new checkpoint selection and do not replace it.
