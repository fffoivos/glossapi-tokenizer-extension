# configs — the frozen machine contracts

> **In one line:** the machine-readable half of this subproject — every scientific and allocation decision that a script had to check, in the state it was left in.
> **Period:** 2026-08-16 → 2026-08-22 (`de6d9b79` … `825e60be`; `phase3_duplicate_exceptions_v1.json` recovered 2026-09-01 in `2aec4a66`). **Status:** complete; several files deliberately retain a pre-execution `status` string.

## Why this existed

The whole study is fail-closed: a script may not infer a scientific value, it must read it from a frozen contract and refuse if the contract is missing, stale or not authorized. These files are those contracts. Their `status` fields are historical and are *not* the completion authority — as [`../HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md`](../HARD_H_TO_G_CROSS_SCALE_EXPERIMENT_HANDOFF_20260822.md) says of the main one, "its historical `status` string predates execution"; the receipts on CSCS are.

## What is here

| File | Role | Recorded status |
| --- | --- | --- |
| [`hard_h_to_g_replication_v1.json`](hard_h_to_g_replication_v1.json) | the authority for the executed study: models, tokenizer, initialization, data, schedule, LR, profiles, evaluation, statistics, launch | `implementation_incomplete_not_launch_authorized` (predates execution) |
| [`legacy_public_greekmmlu_v1.json`](legacy_public_greekmmlu_v1.json) | the pinned legacy BF16 evaluator used for the replication comparison: code revision `cfdd0e7b`, 16,632 questions, max input 3,072, batch 16; also records that the full public panel is the corrected primary | `frozen` |
| [`1p5b_td_acceptance_policy_v2.json`](1p5b_td_acceptance_policy_v2.json) | the architecture-local row-norm band that replaced the 8B-derived band after the 08-15 diagnostic | `proposal_pending_owner_approval` — but the 1.5B run happened; the approval receipt lives on Clariden |
| [`1p5b_tokenizer_compatibility_v1.json`](1p5b_tokenizer_compatibility_v1.json) | the exact 14 content and 18 added-token-record differences between the 1.5B package and the target 148,480 tokenizer | `frozen` |
| [`greekmmlu_query_regeneration_v1.json`](greekmmlu_query_regeneration_v1.json) | the regenerated decontamination queries required after R2 found the historical queries file deleted | `frozen` |
| [`phase3_duplicate_exceptions_v1.json`](phase3_duplicate_exceptions_v1.json) | the 166 byte-identical foreign-replay duplicates excluded from the Phase-3 selection (123,659 tokens) | `frozen` |
| [`hard_h_to_g_allocation_v1.json`](hard_h_to_g_allocation_v1.json) → [`hard_h_to_g_allocation_v3_minimum_defensible.json`](hard_h_to_g_allocation_v3_minimum_defensible.json) | allocation geometry, from unmeasured planning to the 08-18 minimum-defensible 2-node/4-node candidates | `planning_only_unmeasured_1p5b_profile` → `candidate_…_pending_first_allocation_qualification` |
| [`hard_h_to_g_assets_v1.json`](hard_h_to_g_assets_v1.json) | asset inventory spec for the study | — |
| [`experiment_a_recipe.json`](experiment_a_recipe.json) | Experiment A (academic + polytonic mixture) | `planning_pending_release_internal_polytonic_source_audit` — never trained |
| [`experiment_b_recipe.json`](experiment_b_recipe.json) | Experiment B (update-9,536 continuation) | `retired_by_owner_20260812` |
| [`continuation_data_builder_v1.json`](continuation_data_builder_v1.json) | preservation contract for B's two-stage schedule/pool builder: pinned implementation hashes, the proven invocation, the four resulting receipts | `frozen` |
| [`allocation_plan.json`](allocation_plan.json) | the 08-11 A/B allocation plan | `approved` |
| [`owner_authorization.json`](owner_authorization.json) | the 2026-08-11 owner decisions for A and B | `accepted` |
| [`data_runtime_requirements_v1.txt`](data_runtime_requirements_v1.txt), [`td_xfer_runtime_requirements_v1.txt`](td_xfer_runtime_requirements_v1.txt) | exact pinned versions for the immutable AArch64 data runtime and the separate x86_64 `xfer` runtime | — |
| [`greekmmlu_trajectory_sources_20260822.tsv`](greekmmlu_trajectory_sources_20260822.tsv) | the 34-checkpoint source matrix consumed by the trajectory evaluator | — |

## Outcome

`hard_h_to_g_replication_v1.json` and `legacy_public_greekmmlu_v1.json` are the two files a reader needs; a copy of the former is embedded in each report package's `evidence/`. The A/B contracts and the continuation builder contract are preserved so that a future mix experiment can define a new policy ID rather than editing v1 in place.
