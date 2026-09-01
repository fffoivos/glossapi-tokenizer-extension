# tests — the regression net under the fail-closed gates

> **In one line:** eleven pytest modules that keep the contracts, permits and preflights honest; they were edited on almost every day the study ran.
> **Period:** 2026-08-16 → 2026-08-22 (`de6d9b79` … `f85e195e`). **Status:** complete. The focused gate stayed green (`31 passed`, later `54 passed` for the adapter set); the broad legacy suite ended with five known unrelated failures.

## Why this existed

Every gate in this subproject is a refusal: a script must decline to run when a receipt is stale, a bundle is unaudited, a cache is not byte-matched, or a profile drifts from its permit. Those refusals are only trustworthy if they are tested, and the tests had to be updated in lockstep with each new gate — which is why nearly every operational commit in the log touches this directory.

## What is here

| Module | Gates it covers |
| --- | --- |
| [`test_contracts.py`](test_contracts.py), [`test_hard_h_to_g_contracts.py`](test_hard_h_to_g_contracts.py) | static contract validity for the A/B recipes and for the hard-H2G replication contract |
| [`test_canonical_train_adapter.py`](test_canonical_train_adapter.py), [`test_canonical_campaign_contracts.py`](test_canonical_campaign_contracts.py) | the canonical runner adapter and compiled campaign contracts, including the 08-22 `stable_peak` LR axis and the move of four directory arguments to declared environment inputs |
| [`test_canonical_qualification_evidence.py`](test_canonical_qualification_evidence.py), [`test_intentional_torchrun_teardown.py`](test_intentional_torchrun_teardown.py) | profile qualification evidence and the post-checkpoint torchrun teardown accepted as intentional (upstream issue #128) |
| [`test_phase_cache_isolation.py`](test_phase_cache_isolation.py) | that a phase's blend cache is byte-matched or a proven hardlink superset, never silently widened |
| [`test_greekmmlu_evaluation_gates.py`](test_greekmmlu_evaluation_gates.py), [`test_cross_scale_sentinel_authority.py`](test_cross_scale_sentinel_authority.py) | evaluation gates and the sentinel authority (whose calibration was moved to `pre_finalization` on 08-21, `6dd4dd6e`) |
| [`test_matched_study_statistics.py`](test_matched_study_statistics.py) | the pre-registered statistical decision contract |
| [`test_r2_orchestration.py`](test_r2_orchestration.py) | the orchestration remediations demanded by the R2 review |

## Outcome

- The focused adapter/evaluator gate was reported as `31 passed` after the 08-22 scorer changes and `54 passed` after the campaign-contract change ([`../HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md`](../HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md)).
- The broad historical module was left with five unrelated failures — one missing optional `tokenizers` dependency and four stale assertions in older mix-builder/profile/uenv tests. They were recorded rather than papered over: "no unrelated production code was changed to force the broad legacy suite green".
