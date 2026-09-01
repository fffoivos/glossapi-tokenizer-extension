# 06 · clariden — the Slurm launchers

> **In one line:** 73 `sbatch`/`sh` entry points that executed the whole 0.5B screen on CSCS Clariden, from tokenizer overlay to the five-arm campaign to the endpoint waves.
> **Period:** 2026-08-02 → 2026-08-05. **Status:** completed; the campaign they launched finished at update 38,496 in every arm.
> **Came from / led to:** the Python stages in [`../dataset/`](../dataset/), [`../initialization/`](../initialization/), [`../training/`](../training/), [`../evaluation/`](../evaluation/) and [`../production/`](../production/) → these wrappers → run root `/capstor/scratch/cscs/fffoivos/runs/06_dataset_scheduling_experiments/mini_cpt5_20260803T074854Z`.

## Why this existed

Every scientific step had to run on Clariden as a receipt-producing job under the pinned `pytorch/v2.9.1:v2` uenv, with explicit partition and node choices and no hidden defaults. These scripts are the thin, auditable boundary between the repository's Python and the scheduler; several of them exist only to make a resource decision explicit (a four-lane evaluation node instead of nested one-GPU batch jobs, for example).

## The stages, in the order they ran

| Stage | Entry points |
|---|---|
| Tokenizer overlay + base model | `freeze_base_mini_model.sbatch`, `build_mini_fvt_benchmark_init.sbatch`, `build_token_byte_lengths.sbatch` |
| Pool freeze and validation | `submit_initial_data_build.sh`, `prepare_data_mixes.sbatch`, `validate_partition_group.sbatch`, `submit_pool_validation.sh`, `finalize_pool_corpus.sbatch`, `diagnose_modern_content_duplicates.sbatch` |
| Packing and the five schedules | `build_packing_plan.sbatch`, `pack_catalog_bucket.sbatch`, `submit_dynamic_packing.sbatch`, `finalize_packed_corpus.sbatch`, `submit_packing_and_schedules.sh`, `build_five_schedules.sbatch`, `build_train_shards.sbatch` |
| Tied Token Distillation | `submit_tied_td_pipeline.sh`, `run_tied_td_pilots.sbatch`, `evaluate_tied_td_pilots.sbatch`, `run_full_tied_td.sbatch`, `verify_full_tied_td.sbatch`, `finalize_tied_td_initialization.sbatch`, `build_packed_training_td_assets.sbatch`, `convert_full_td_to_megatron.sbatch`, `finalize_td_conversion_receipt.sbatch`, `build_production_megatron.sbatch` |
| Validation panels | `build_heldout_shards.sbatch`, `build_validation_manifest.sbatch`, `submit_neutral_external_pipeline.sh`, `prepare_neutral_external_source.sbatch`, `build_neutral_candidate_signatures.sbatch`, `match_neutral_minhash_bucket.sbatch`, `finalize_neutral_cross_dedup.sbatch`, `build_neutral_external_heldout.sbatch`, `build_lm_eval_runtime.sbatch` |
| Learning-rate smoke | `run_common_stability_smoke.sbatch` (candidate `3e-4`), `run_common_stability_fallback.sbatch` (accepted `1.5e-4`), `recover_common_stability_endpoint.sbatch` |
| Prelaunch | `run_five_arm_prelaunch_smoke.sbatch`, `run_prelaunch_smoke_arm.sh`, `finalize_static_prelaunch_evidence.sbatch`, `finalize_prelaunch_campaign.sh` |
| The campaign | `submit_production_campaign.sh`, `run_initial_validation.sbatch`, `train_five_arm_segment.sbatch`, `run_production_arm_segment.sh`, `supervise_production_segment.sbatch`, `gate_segment.sbatch`, `freeze_segment_checkpoint.sbatch`, `status_production_campaign.sh` |
| Checkpoint → GreekMMLU service | `build_checkpoint_evaluation_plan.sbatch`, `convert_checkpoint_for_native_greekmmlu.sbatch`, `finalize_checkpoint_export.sbatch`, `run_checkpoint_native_greekmmlu{,_one.sh,_wave}.sbatch`, `submit_checkpoint_native_greekmmlu{,_wave}.sh`, `finalize_checkpoint_greekmmlu.sbatch`, `watch_checkpoint_evaluations.sbatch`, `run_checkpoint_evaluation_backlog_{batch,controller}.sbatch`, `accelerate_checkpoint_evaluation_backlog.sbatch` |
| Evaluation-precision diagnosis | `diagnose_native_greekmmlu_dtype.sbatch`, `diagnose_native_greekmmlu_dtype_wave.sbatch` |
| Endpoints and closure | `run_full_endpoint_validation.sbatch`, `run_greek_endpoint_wave.sbatch`, `run_retention_endpoint_{one.sh,shard.sbatch,wave.sbatch}`, `finalize_retention_endpoint_shards.sbatch`, `finalize_core_campaign_evidence.sbatch`, `finalize_campaign_evidence.sbatch` |

## Two decisions embedded in these scripts

- **`run_checkpoint_native_greekmmlu_wave.sbatch` replaced nested one-GPU batch jobs.** Slurm accounted those as complete four-GPU nodes on Clariden. The replacement follows CSCS's documented node-sharing pattern: one exclusive four-GPU allocation, four resource-isolated one-GPU `srun --exclusive` steps with 64 cores and 105 GB each, then the fifth arm on the first freed lane. This was proven by the four-lane wave smoke (job `2983668`, five exact receipts in 516 s — `../evidence/native_greekmmlu_four_lane_wave_smoke_20260802.json`).
- **`status_production_campaign.sh` is read-only.** It is designed to be streamed to Clariden over `ssh` from a laptop without touching the frozen scientific bundle or the campaign run root: it reports the submission graph, Slurm states, immutable receipts, latest per-arm training/checkpoint iterations and any fatal or non-finite diagnostics.

## Outcome

- The campaign launched on 2026-08-03 with jobs `2989297` (initial validation), `2989298` (five-arm training on 20 nodes / 80 GPUs), `2989299` (evaluation watcher) and `2989300` (segment supervisor), and advanced itself from verified common checkpoints thereafter.
- `train_five_arm_segment.sbatch` is the script subproject 07 inherited conceptually; its 8B counterpart is [`../../07_full_8b_cpt/clariden/train_segment.sbatch`](../../07_full_8b_cpt/clariden/train_segment.sbatch).

## Working documents

All 73 files are historical launchers. None is superseded within the directory; the `*_wave*` and `*_backlog*` variants coexist with their single-job forms because the campaign used both (steady-state waves, plus bounded backlog drains when evaluations fell behind training).
