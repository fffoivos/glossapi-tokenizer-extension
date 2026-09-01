# scripts — builders, freezers, auditors, preflights

> **In one line:** 136 Python modules that do the actual work behind the Slurm jobs — every receipt, permit, gate and contract in this study is written by something in here.
> **Period:** 2026-08-16 → 2026-08-22, plus two `rebind_*.py` recovered on 2026-09-01 (`2aec4a66`). **Status:** complete.

## Why this existed

The subproject's design rule is that no scientific value may be inferred at run time: it is derived once, written into an immutable receipt with hashes, and thereafter only verified. That makes most of this directory pairs — a *builder* that derives something and an *auditor* or *verifier* that independently proves it — with a *freezer* in between that makes the result immutable and binds it to the exact code bundle that produced it.

## How to read it

- **`build_*`** — derive an artifact: source views, document catalogs, phase data-path specs and GPTDataset caches, checkpoint permits, launch and operational gates, campaign contracts, frozen GreekMMLU queries, the retired Experiment-B schedule and pool view.
- **`freeze_*`** — make it immutable and bind it: experiment contracts, artifact manifests, owner authorization, producer-bundle compatibility, tokenized streams, blend caches, LR selection, TD row-norm contract, the statistical decision contract, the stable-peak branch gate, production timing and allocation.
- **`audit_*` / `verify_*` / `inspect_*`** — independent checks: decontamination and validation-exclusion audits, packed-payload integrity, training checkpoints, TD initialization, the data runtime and the `xfer` runtime, DCP metadata compatibility, restart checkpoint tensors and metadata.
- **`preflight_train_segment.py`** — the gate every production segment passes: Phase 1/2 needs `pre_main`, Phase 3 needs `pre_extension` then `pre_second_extension`, so a bare `sbatch` cannot bypass the staged manifest or the owner authorization.
- **`run_*`** — in-job drivers: `run_canonical_train_segment.py`, `run_in_allocation_profile_qualification.py`, `run_parallel_task_batch.py`, `run_fresh_greekmmlu_stream_scan.py`, `run_legacy_greekmmlu_snapshot_eval.py`.
- **`patch_*` / `promote_*` / `materialize_*`** — runtime and bundle plumbing: uenv-v10 `srun` and scale-geometry patches, canonical runtime promotion, sparse scientific-bundle overlays, pinned HF model materialization.
- **`workaround_*` and `rebind_*`** — named, tested deviations kept visible rather than folded into the main path: accepting an intentional torchrun teardown (upstream issue #128), building the current S2 continuation, parameterized profile qualification, rebinding a resized continuation contract, rebinding the pre-authorization manifest and the training-authorization gate.
- **Retired Experiment B** — `build_continuation_b_schedule.py` and `build_continuation_b_pool_view.py`, preserved under the contract in [`../CONTINUATION_DATA_BUILDER_HANDOFF_20260812.md`](../CONTINUATION_DATA_BUILDER_HANDOFF_20260812.md): their v1 files must not be edited in place for a different data mix; a new mix needs a new policy ID.

## Outcome

- The gate chain worked as designed at least twice under pressure: a stale training-run permit was correctly rejected after its bundle changed, and a v9 branch gate correctly rejected a v8 checkpoint permit rather than accepting a copied one ([`../HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md`](../HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md)).
- Fifteen data wrappers that still defaulted to a known-incomplete PyArrow environment were made fail-closed on an explicit `H2G_DATA_PYTHON` during R2 remediation, even though the runs that used them had overridden the default correctly ([`../ULTRACODE_R2_REMEDIATION_20260814.md`](../ULTRACODE_R2_REMEDIATION_20260814.md)).
- `diagnose_1p5b_td_row_norms.py` produced the one diagnostic that changed a scientific decision here — see [`../1P5B_TD_ROW_NORM_DIAGNOSTIC_20260815.md`](../1P5B_TD_ROW_NORM_DIAGNOSTIC_20260815.md).
