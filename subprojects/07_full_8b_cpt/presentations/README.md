# 07 · presentations — the reports, in the order the run produced them

> **In one line:** eleven self-contained HTML reports with frozen JSON payloads, tracing the 8B run from a stopped exploratory prefix, through mid-run progress, to the final trajectory and the drift investigation that followed it.
> **Period:** 2026-08-06 → 2026-08-12. **Status:** completed; three of these were promoted to subproject 09 as canonical, the rest are working evidence.
> **Came from / led to:** campaign receipts in the run root → these builders → [`../../09_full_8b_cpt_results_analysis/presentations/`](../../09_full_8b_cpt_results_analysis/presentations/).

## The sequence

| Date | Report | What it said |
|---|---|---|
| 08-06 | `FULL8_EXPLORATORY_PREFIX_20260806` | The **stopped** pre-anonymization trajectory, through update 7,152 = 29,997,662,208 token slots (37.2% of the planned horizon). Greek learning is broad, not confined to GlossAPI; foreign panels show small departures from their running minima; GreekMMLU rose 35.28% → 56.58% (update 5,960) and oscillated to 55.24%, on the full 16,632-question set. Explicitly "a useful trajectory, not a reusable endpoint" — the run was stopped because the text had not had the required PII pass. |
| 08-09 | `FULL8_SANITIZED_RERUN_PROGRESS_20260809` | Mid-run status of the sanitized rerun: "learning replicates better than the benchmark". Same model, different corpus realization; training numerically stable; absolute BPB keeps improving; a small upward foreign drift; GreekMMLU not following validation loss monotonically. Attributes the difference primarily to the changed sample trajectory and secondarily to horizon rescaling — not to architecture. |
| 08-10 | `FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260810` | First cross-scale comparison. **Superseded the next day.** |
| 08-11 | `FULL8_SANITIZED_CPT_FINAL_RESULTS_20260811` | The canonical final report. Complete trajectory, all 13 learning/forgetting panels, exact document-local endpoints, all 19 GreekMMLU checkpoints. Accuracy +19.07 points from initialization; best checkpoint update 9,536 at 56.81% / 1.0740 NLL / 0.1701 answer BPB; endpoint −1.96 points below that. Cooldown improves all 13 exact panels. Completion receipt written 10:49 Athens, binding 19 GreekMMLU + 39 document-local receipts, 5 training-attempt audits, the launch gate, the DP32 profile and the terminal export. |
| 08-11 | `FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260811` | The 0.5B screen supports D0 but does not guarantee monotonic 8B benchmarks; separates what replicated conceptually from what cannot be attributed. |
| 08-11 | `FULL8_GREEKMMLU_DRIFT_AND_DATA_EXPOSURE_20260811` (+ `../output/pdf/…pdf`) | Checkpoint answer drift against source exposure. |
| 08-11 | `GREEKMMLU_HISTORICAL_DRIFT_FRAMEWORK_20260811`, `GREEKMMLU_DRIFT_SIMULATION_STRESS_TEST_20260811`, `GREEKMMLU_HISTORICAL_CHANGE_SIMULATIONS_20260811`, `GREEKMMLU_WRONG_CELL_DISPLACEMENT_BAKEOFF_20260811`, `GREEKMMLU_DRIFT_HISTORY_GALLERY_20260811` | The statistic-selection work behind [`../analysis/GREEKMMLU_RESPONSE_DISPLACEMENT_BY_CATEGORY_20260811.md`](../analysis/GREEKMMLU_RESPONSE_DISPLACEMENT_BY_CATEGORY_20260811.md): synthetic drift simulations, a stress test, historical-change simulations and a bake-off between candidate displacement statistics. Subproject 09 deliberately did **not** copy these forward — they did not change any decision. |

## Builders

Each report has a builder next to it (`build_exploratory_prefix_report.py`, `build_current_vs_previous_report.py`, `build_full8_standalone_report.py` + `build_full8_final_results_report.py`, `build_full8_vs_five_arm_report.py`, `build_checkpoint_answer_drift_report.py`, `build_greekmmlu_drift_simulation_report.py`, `build_greekmmlu_historical_change_simulations.py`, `build_greekmmlu_wrong_cell_displacement_report.py`, `build_greekmmlu_drift_history_gallery.py`). Every report ships its own `.data.json` payload with the source receipt paths and hashes it was built from, so the numbers are checkable without CSCS access.

## Outcome

- `FULL8_SANITIZED_CPT_FINAL_RESULTS`, `FULL8_VS_0P5B_FIVE_ARM_COMPARISON` (08-11 build) and the drift/exposure report were promoted into [`../../09_full_8b_cpt_results_analysis/`](../../09_full_8b_cpt_results_analysis/) as canonical, where they are listed in `evidence/ARTIFACT_MANIFEST.json` with their hashes.
- The exploratory prefix report is the reason the pre-sanitization trajectory can be discussed at all; it is evidence, not a result.

## Working documents

Historical, all of it. Superseded within the directory: `FULL8_VS_0P5B_FIVE_ARM_COMPARISON_20260810.*` by the `20260811` build. The five GreekMMLU drift/simulation reports are exploratory analyses that no decision depended on.
