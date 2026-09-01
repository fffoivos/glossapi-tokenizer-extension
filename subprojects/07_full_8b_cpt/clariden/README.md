# 07 · clariden — the Slurm entry points

> **In one line:** 76 launchers covering the whole 8B campaign — data graph, anonymization, benchmarks and smokes, the five training segments, the serial debug evaluation chain, and the read-only watchers — organised around one hard constraint: `normal` is only ever for 16-node training.
> **Period:** 2026-08-05 → 2026-08-09. **Status:** completed.
> **Came from / led to:** the Python stages in [`../dataset/`](../dataset/), [`../evaluation/`](../evaluation/), [`../train/`](../train/) and [`../scripts/`](../scripts/) → these wrappers → run root `20260808T121000Z-d0-wsd10-sanitized-successor-v12`.

## The resource rule

From [`../FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md`](../FULL8B_RERUN_LAUNCH_HANDOFF_20260808.md): `debug` takes every short one-node metadata, receipt, launch-gate, control, conversion, GreekMMLU and per-document job that fits the 90-minute limit; `normal` takes only the five 16-node training segments. No 16-node request is submitted until every allocation-free and debug gate is green, and `sbatch --test-only` is a prediction, never a reservation. Live limits observed 2026-08-08: `debug` max 4 nodes / 01:30:00; `debug-qos` **one running and two submitted jobs per user**; `normal` 12 hours. Three consequences are baked into these scripts — a debug controller must never wait on debug children (it would consume the only running slot), the evaluation graph must be a serial chain rather than a prequeued fan-out, and every partition choice is passed explicitly on the command line because script-header defaults are not trusted.

## Groups

| Stage | Entry points |
|---|---|
| Data graph | `submit_data_pipeline.sh`, `freeze_data_inventory.sbatch`, `pack_full_data.sbatch`, `finalize_data.sbatch`, `verify_packed_payload_hashes.sbatch` |
| Anonymization | `submit_anonymization_pipeline.sh`, `freeze_anonymization_overlay.sbatch`, `build_anonymization_inventory_group.sbatch`, `promote_compatible_inventory.sbatch`, `build_sanitized_binary_group.sbatch`, `finalize_postmask_dedup.sbatch`, `finalize_sanitized_bridge.sbatch`, `smoke_anonymization_pipeline.sbatch` |
| Validation rebuild | `build_all_panel_validation_overlay.sbatch`, `build_corrected_validation_stage.sbatch`, `materialize_corrected_initial_hf.sbatch`, `freeze_initial_checkpoint.sbatch` |
| Benchmarks and smokes | `submit_parallelism_benchmark.sh`, `finalize_parallelism_benchmark.sbatch`, `submit_checkpoint_parity_smoke.sh`, `finalize_checkpoint_parity_smoke.sbatch`, `submit_dp32_fallback_selection.sh`, `finalize_dp32_fallback_selection.sbatch`, `submit_graceful_stop_smoke.sh`, `signal_graceful_stop_smoke.sbatch`, `resume_graceful_stop_smoke.sbatch`, `wait_graceful_stop_receipt.sbatch`, `finalize_graceful_stop_smoke.sbatch`, `submit_conversion_smoke.sh`, `finalize_conversion_smoke.sbatch`, `prove_nested_sbatch.sbatch`, `nested_sbatch_child.sbatch`, `wait_nested_sbatch_proof.sbatch` |
| Launch | `submit_prelaunch_evaluations.sh`, `start_sanitized_prelaunch_after_benchmark.sbatch`, `submit_sanitized_prelaunch_and_launch.sh`, `submit_corrected_prelaunch_and_launch.sh`, `finalize_and_submit_production.sbatch`, `submit_production.sh` |
| Training | `train_segment.sbatch`, `supervise_campaign.sbatch`, `freeze_checkpoint.sbatch`, `finalize_campaign.sbatch`, `resolve_leaf_switch_exclusion.sh` |
| Resource-aware layer (08-08/09) | `deploy_resource_aware_bundle.sh`, `submit_production_resource_aware.sh`, `finalize_and_submit_production_resource_aware.sbatch`, `supervise_campaign_resource_aware.sbatch`, `prove_resource_aware_routing.sbatch`, `resource_aware_routing_child.sbatch`, `prequeue_next_segment_debug.sbatch`, `run_prequeued_train_holder.sbatch`, `prepare_successor_stage_debug.sbatch`, `prepare_successor_contracts_debug.sbatch`, `prove_successor_launch_gate_debug.sbatch` |
| Evaluation chain | `run_initial_greekmmlu.sbatch`, `run_checkpoint_source_validation.sbatch`, `run_evaluation_queue.sbatch`, `run_checkpoint_evaluation_debug.sbatch`, `continue_checkpoint_evaluation_debug.sbatch`, `finalize_split_checkpoint_evaluation_debug.sbatch`, `run_per_document_group{.sbatch,_debug.sbatch,_resource_aware.sh}`, `submit_per_document_validation.sh`, `prove_evaluation_overlap.sbatch` |
| 0.5B per-document rerun (prepared, dry-run by default) | `submit_mini_per_document_rerun.sh`, `build_mini_per_document_manifest.sbatch`, `run_mini_per_document_smoke.sbatch`, `finalize_mini_per_document_comparison.sbatch` |
| Post-hoc analysis | `analyze_greekmmlu_drift_and_exposure.sbatch`, `analyze_greekmmlu_response_displacement.sbatch`, `extract_greekmmlu_history_matrix.sbatch` |
| Read-only watchers | `watch_full8b_campaign.sh`, `watch_pending_supervisor_transition_v29.sh` |

## Three decisions embedded here

- **Placement.** Multi-node training requires `--switches=1`, excludes every node outside a predeclared leaf group (`resolve_leaf_switch_exclusion.sh`), and verifies the actual allocation against Clariden's live leaf-switch topology before model setup. The v36 parity attempt spread over six leaf switches showed 8.7–72.0 second updates against a proven ≈8.74 s single-switch trajectory (`85f2b755`, `c7652b1e`).
- **uenv nesting.** `srun` starts from the ordinary batch environment and the pinned uenv is mounted once inside each node-local rank launcher. Wrapping `srun` in `uenv run` and calling `uenv run` again inside its tasks is invalid on Clariden — the v34 DP32 control failed before its first optimizer update with an explicit "a uenv session is already running" guard (`61704c0a`, `98d3bd27`, `f2f905f8`).
- **Handoff, not a button.** `finalize_and_submit_production.sbatch` refreshes live storage and scheduler evidence, builds the sole launch gate from the complete receipt set, executes the dry run, and only then calls the real submitter — requiring the literal `APERTUS8B_FULL_MIXED_CPT` authorization value and refusing to overwrite an existing environment receipt, launch gate or run root (`9b2433d5`).

## Watchers are read-only by construction

`watch_full8b_campaign.sh` runs as a local LaunchAgent and only records the Slurm graph, the newest training-log marker, the latest loss/health line and the last stderr line every two minutes, exiting when the terminal training receipt appears. `watch_pending_supervisor_transition_v29.sh` is a finite six-hour macOS-side coordinator for exactly one supervisor swap. Neither has any `sbatch`, `scancel`, remote-write, GPU or data action.

## Working documents

Seventy-six files. `train_segment.sbatch.orig` is a retained pre-patch copy of the training launcher, kept for diffing; the live script is `train_segment.sbatch`. The `*_resource_aware*` and `*_debug*` variants are the 2026-08-08/09 operational layer that calls the immutable v45 scientific scripts with explicit routing — they supplement rather than replace their originals.
