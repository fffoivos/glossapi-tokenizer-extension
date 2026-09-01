# presentations — the three report packages

> **In one line:** the study reported itself three times in two days, each time on a wider question set: endpoint-only, then the full matched trajectory, then the corrected full public panel plus the no-decay branch.
> **Period:** 2026-08-22 → 2026-08-23 (`de52e1bc`, `36af6f13`, `75e4077b`; the endpoint package was recovered on 2026-09-01 in `2aec4a66`). **Status:** complete — the third package is the final result.

## Why this existed

Each package is a self-contained, rebuildable HTML report with its own `evidence/` snapshot, a builder, a QA verifier and rendered screenshots at desktop and narrow widths. The pattern matters because the primary metric changed mid-flight: rather than editing one report in place, each correction produced a new package whose builder fails closed if its evidence is missing, so the superseded numbers stay readable next to the ones that replaced them.

## The sequence

| Dir | Date | Question it answered | Headline |
| --- | --- | --- | --- |
| [`greekmmlu_endpoint_20260822/`](greekmmlu_endpoint_20260822/) | 2026-08-22 | at the final checkpoint only, how do the two scales compare? | 8B 9,232/16,159 = 57.13%, 1.5B 6,429/16,159 = 39.79% (16,159-question subset) |
| [`greekmmlu_trajectory_20260822/`](greekmmlu_trajectory_20260822/) | 2026-08-22 | does the 1.5B *trajectory* mirror 8B across all 17 checkpoints? | no — HPLT→OA delta −3.09 pp vs +1.27 pp; level correlation −0.6698 |
| [`hard_h2g_full_panel_stable_lr_20260822/`](hard_h2g_full_panel_stable_lr_20260822/) | 2026-08-23 | on the corrected full public panel, was the historical result replicated, and does a constant LR keep the curve rising? | replication miss (best legacy-BF16 57.9004% vs 59.9627%); the no-decay arm falls |

The second superseded the first within the day; the third superseded the second's *metric* (the 16,159-question subset was demoted to a sensitivity analysis after it was recognised as a second application of decontamination) without invalidating its cross-scale conclusion.

## Outcome

The third package is the final deliverable of subproject 08: a 135,361-byte single page (SHA-256 `6e4ae4a6…`) built from a 661,953-byte `evidence/analysis.json`, with a passing QA receipt. Review integration was opened as `train-apertus-with-glossapi` PR #17. Note that only [`hard_h2g_full_panel_stable_lr_20260822/README.md`](hard_h2g_full_panel_stable_lr_20260822/README.md) is original to the study — it is a build recipe and is left as written.
