# 06 · presentations — the result reports

> **In one line:** the canonical five-arm result report and its frozen payloads, plus two learning-rate reports that belong to the 8B track but explain why this experiment held WSD-10 fixed.
> **Period:** 2026-08-01 → 2026-08-05. **Status:** completed; `DATA_ORDER_MIX_RESULTS_20260805.html` is the canonical output of subproject 06.
> **Came from / led to:** [`../evaluation/`](../evaluation/) receipts → this → the D0 decision in [`../../07_full_8b_cpt/`](../../07_full_8b_cpt/) and the scale comparison in [`../../09_full_8b_cpt_results_analysis/`](../../09_full_8b_cpt_results_analysis/).

## Contents and history

### `DATA_ORDER_MIX_RESULTS_20260805.html` — the result (2026-08-05)

Built by [`build_dataset_order_results_report.py`](build_dataset_order_results_report.py) from the five frozen payloads in [`data/dataset_order_20260805/`](data/dataset_order_20260805/): `dataset_order_selection_analysis.json`, `core_campaign_summary.json`, `greekmmlu_trajectory.json` (415 rows), `validation_trajectory.json` (5,395 rows) and `full_endpoint_validation_receipt.json` (65 rows). It shows every raw checkpoint measurement from initialization to update 38,496, with no smoothing and no checkpoint averaging.

What it reports:

- **The tradeoff.** D1 (hard HPLT→GlossAPI) finishes best on every curated GlossAPI family but pays +0.0473 BPB on HPLT and +0.0184 on neutral Greek versus D0. D2 (hard GlossAPI→HPLT) finishes best on HPLT but gives back +0.0823 BPB on aggregate non-HPLT and +0.0855 on historical polytonic.
- **Replay.** D0 has the lowest observed replay-forgetting macro (0.09475 BPB), D3 essentially adjacent (0.09487). "Forgetting" is defined as final fast-panel loss minus that arm's *best* fast-panel loss anywhere in training, not merely movement during cooldown.
- **Benchmarks.** On the 16,159-question decontaminated subset: D3 leads accuracy at 42.37%, D0 leads final choice NLL at 1.2869, D2 leads correct-answer BPB at 0.1856. D3's +0.24 pp accuracy lead has a paired 95% interval of −0.22 to +0.71 pp.
- **The honest ending.** D0 is first under the stated hierarchy after the 5% source-retention safety screen and its GreekMMLU choice-NLL advantage is paired-question robust — but formal winner selection stays blocked because the frozen validation receipts store one aggregate per panel rather than document-cluster numerators, and no pre-endpoint numeric margin exists for the general benchmark suite.

### `LR_SCHEDULES_AS_RUN_AND_NEXT_20260801.html` and `LR_SCHEDULE_TAIL_EXPERIMENTS_20260801.pptx` (2026-08-01)

Reports on the **8B** learning-rate work, not this 0.5B screen: the completed sweep changed the peak rather than the tail rule, `5.5e-5` was the practical knee for the 8B pilot, and — the conclusion this subproject acted on — for Mini one should transfer the tested *fraction*, not the 8B absolute number. That reasoning produced the `3e-4` candidate that the stability smoke later rejected. The `.pptx.inspect.ndjson` sidecar is a slide-inspection dump.

### `LR_FLOOR_EXPERIMENT_RESULTS_20260802.html` (2026-08-02)

The 8B T10/T20/T30 learning-rate-floor study. Its own summary is that the valid comparison is complete but endpoint selection is not, and it notes that reconstructed comparable shapes do not reproduce the previously reported losses. This is why both [`../FACTORIAL_EXPERIMENT_DESIGN.md`](../FACTORIAL_EXPERIMENT_DESIGN.md) §4 and the 8B recipe treat WSD-10 as a **settled baseline rather than a sweep-selected winner**, and why the 10/20/30% floor study was explicitly deferred out of this round.

## Outcome

- One canonical report with locally readable payloads — the five-arm result can be re-derived on a laptop without CSCS access.
- The two LR reports are the documented reason the screen has exactly one LR arm.

## Working documents

Everything here is a finished artifact. `data/dataset_order_20260805/` retains the original CSCS receipt paths and hashes inside the payloads; the run root they came from is `/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/mini_cpt5_20260803T074854Z`.
