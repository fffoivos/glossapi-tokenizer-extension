# clariden — the Slurm entry points

> **In one line:** 153 `sbatch`/shell entry points covering everything this study ran on CSCS Clariden, from the first read-only HF inspection to the segmented trainer and the checkpoint audit.
> **Period:** 2026-08-16 → 2026-08-22 (all committed in `de6d9b79` and the operational commits that followed). **Status:** complete; the study finished and no further jobs were submitted from here after 2026-08-23.

## Why this existed

Two rules from [`../CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md`](../CPT_EXPERIMENT_AND_RESOURCE_PLAN_20260811.md) explain the shape of this directory. First, resource discipline: everything that is not actual training or a bounded distributed parity smoke runs on a one-node `debug` allocation or the allocation-free login path — hence the `_debug.sbatch` suffix on most files. Second, fail-closed staging: each step reads the previous step's immutable receipt and refuses to run without it, so the pipeline is expressed as many small jobs rather than one script.

## How to read it

The filenames encode the stage. Roughly in execution order:

- **Inspect and inventory** — `inspect_hf_release_debug`, `inspect_replay_parquet_debug`, `inspect_native_overlap_schema_debug`, `inventory_hard_h_to_g_assets_debug`.
- **Build the immutable runtimes and bundle** — `deploy_targeted_bundle.sh`, `build_data_runtime_debug`, `build_td_xfer_runtime` (the separate x86_64 CPU-only `xfer` runtime), `validate_and_inspect_debug`, and the nested-submission and uenv proofs `prove_nested_sbatch_debug`, `prove_nested_same_uenv_srun_debug`, `prove_uenv10_srun_debug`.
- **Select, decontaminate, exclude** — `extract_hplt_candidates`, `extract_academic_sources`, `extract_release_polytonic_sources`, `decontaminate_*`, `exclude_*_validation`, `audit_*` and `filter_replay_*`, plus the `search_*_poly_artifact` jobs from the 08-12 detour that was superseded once the polytonic sources were taken directly from the pinned release.
- **Tokenize, split, cache** — `split_replay_stage_b_debug`, `tokenize_h2g_stream_debug`, `tokenize_phase3_stream_debug`, `build_phase_data_path_spec_debug`, `build_phase_gptdataset_cache_debug`, `freeze_phase_blend_cache_debug`, `materialize_phase_cache_debug`.
- **Initialize** — `build_1p5b_td_init_debug`/`_normal`, `build_td_snippets_*`, `roundtrip_td_init_debug`, `diagnose_1p5b_td_row_norms_debug` (the 08-15 diagnostic), `freeze_td_norm_contract_debug`, `freeze_1p5b_td_policy_authorization_debug`, `materialize_historical_tokenizer_debug`.
- **Gate and authorize** — the `freeze_*` family: artifact manifests (`freeze_pre_main_artifact_manifest_debug`, `freeze_extension_artifact_manifest_debug`), owner authorization, launch gates, producer-bundle compatibility, production timing and allocation, profile promotion, training-run permits.
- **Train** — [`train_hard_h_to_g_segment.sbatch`](train_hard_h_to_g_segment.sbatch) (the fail-closed segmented trainer, gated by `../scripts/preflight_train_segment.py`), `run_targeted_restart_smoke.sbatch` / `submit_targeted_restart_smoke.sh`, `submit_targeted_production.sh`, `run_phase3_resume_smoke.sbatch`, `run_prelaunch_benchmark.sbatch`.
- **Evaluate and audit** — `export_checkpoint_for_evaluation_debug`, `run_frozen_greekmmlu_4node_debug`, `run_legacy_public_greekmmlu_debug`, `run_offline_panels_4node_debug`, `run_all_per_document_groups_debug` (all 13 panels in one four-GPU allocation), `run_native_suite_checkpoint_4node_debug`, `audit_training_checkpoint_debug`.
- **Retired Experiment B** — `build_continuation_b_schedule_debug`, `build_continuation_b_pool_view_debug`, `prepare_continuation_b_assets_debug`, `finalize_and_submit_b_after_restart_debug`. Kept as immutable machinery; B was never launched.

`hard_h_to_g_training.env` holds the shared training environment and `verify_data_runtime.inc.sh` the shared runtime-verification snippet.

## Outcome

- The debug-only discipline held: apart from training and the bounded parity smoke, the study's preparation, gating, conversion and evaluation ran on one-node `debug` or on the login path. The one measured exception is the 2B-token TD coverage scan, which was moved to the CPU-only `xfer` partition after a 2026-08-15 measurement (50,002,836 tokens in 7 m 48 s, ≈5.2 h projected) proved it could not fit the 90-minute debug limit.
- Several entry points were superseded in flight rather than deleted: `deploy_targeted_bundle.sh` could no longer build from the 08-22 worktree (it requires a training patch that had left the repository revision), so the final scoring bundle was assembled server-side from the already proven evaluator bundle — see [`../HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md`](../HARD_H2G_FULL_PANEL_AND_NO_DECAY_EXECUTION_NOTES_20260822.md), "Failures and reusable findings".
- Where a frozen entry point could not be re-run inside a held allocation, the adapter went to [`../operational_workarounds/`](../operational_workarounds/) rather than into these files.
