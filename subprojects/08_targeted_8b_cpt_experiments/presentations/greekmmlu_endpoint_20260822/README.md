# greekmmlu_endpoint_20260822 — endpoint-only cross-scale report

> **In one line:** the first report of the matched study, comparing only the two final (update-3,694) checkpoints; superseded within the day by the full trajectory report.
> **Period:** 2026-08-22 (recovered from an uncommitted working tree on 2026-09-01, `2aec4a66`). **Status:** superseded by [`../greekmmlu_trajectory_20260822/`](../greekmmlu_trajectory_20260822/) and then by [`../hard_h2g_full_panel_stable_lr_20260822/`](../hard_h2g_full_panel_stable_lr_20260822/).

## Why this existed

As soon as both trajectories reached update 3,694, the endpoint scores existed while the per-checkpoint trajectory scoring was still running. This package reported the endpoint comparison first.

## What it says

On the frozen 16,159-question GreekMMLU subset at update 3,694: 8B `9,232/16,159 = 57.132%`, choice NLL `1.0800`, correct-answer BPB `0.17480`; 1.5B `6,429/16,159 = 39.786%`, choice NLL `1.3467`, BPB `0.20079` ([`evidence/8b_summary.json`](evidence/8b_summary.json), [`evidence/1p5b_summary.json`](evidence/1p5b_summary.json)). The historical curve drawn alongside is from [`evidence/historical_cpt_2arm_summary.json`](evidence/historical_cpt_2arm_summary.json) — the earlier surviving two-arm TD study, **not** the selected β₂ arm that is the formal replication target.

## Why it was superseded

An endpoint-only comparison cannot answer the mirroring question: the 1.5B model's best checkpoint is at update 1,904, far before the endpoint, and its accuracy declines through the OpenArchives phase while its choice NLL improves. Both facts are only visible across the full trajectory, which is why [`../greekmmlu_trajectory_20260822/`](../greekmmlu_trajectory_20260822/) replaced this package the same day. The 16,159-question panel it uses was itself later demoted to a sensitivity analysis in favour of the complete 16,632-question public panel.

## Contents

`GREEKMMLU_H2G_CROSS_SCALE_RESULTS_20260822.html`, its `build_report.py`, `evidence/` (the two endpoint summaries, the historical two-arm curve, `training_trajectories.json`, and a copy of the replication contract) and `qa/` (desktop and narrow renders, in draft and final form).
