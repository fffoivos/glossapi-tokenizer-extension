#!/usr/bin/env bash
set -euo pipefail
[[ "${SLURM_JOB_ID:-}" == 3146990 && "${SLURM_JOB_PARTITION:-}" == normal && "${SLURM_NNODES:-0}" == 2 ]] || exit 2
ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
CODE=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T014000Z-hard-h2g-ledger-invariant-v15
RUN1="$ST/runs/h2g_1p5b_2node_v1_retry14_direct_salloc_uenv_20260821"
export H2G_CODE_ROOT="$CODE" H2G_CODE_RECEIPT="$CODE.receipt.json" H2G_STAGE_ROOT="$ST"
export H2G_MODEL_SCALE=1p5b H2G_PHASE=3 H2G_START_UPDATE=3218 H2G_EXIT_UPDATE=3456
export H2G_LOAD_CHECKPOINT="$RUN1/segments/s3/attempts/attempt_000002/payload/checkpoints" H2G_CHECKPOINT_PERMIT="$RUN1/segments/s3/attempts/attempt_000002/checkpoint_permit.json"
export H2G_SOURCE_PHASE_CACHE_RECEIPT="$ST/receipts/phase_2_blend_cache_1p5b_val148_v116_v10_postprocess_rebind.json"
export H2G_PHASE_DATA_PATH_SPEC="$ST/data/phases/phase3/phase_data_path.json"
export H2G_PHASE_DATA_PATH="1.0 $ST/megatron/phase3_openarchives_ext_text_document 0.253164557 $ST/megatron/phase3_foreign_ext_text_document 0.012658228 $ST/megatron/phase3_old_greek_ext_text_document"
export H2G_PHASE_CACHE_RECEIPT="$ST/receipts/phase_3_blend_cache.json" H2G_PHASE_CACHE_ROOT="$ST/data/phases/phase3/cache" H2G_PHASE_CACHE_TREE_SHA256=6575b7e478a1db225facb2d2c2ea6edc62e7dc7cfb0a6a5d9d7b56698f1d2260
export H2G_RUNTIME_PHASE_CACHE_ROOT="$ST/cache_overlays/phase3_8b_historical_validation_v14" H2G_CACHE_OVERLAY_RECEIPT="$ST/receipts/phase_3_overlay_8b_historical_validation_v14.json"
export H2G_PRODUCER_COMPATIBILITY="$ST/receipts/producer_bundle_compatibility_extension_gate_v15_phase3_ledger.json"
export H2G_SEGMENT_OUTPUT_ROOT="$ST/runs/direct_v15_1p5b_s4_3218_3456_3146990"
export H2G_PEAK_LR=5.5e-5 H2G_FLOOR_LR=5.5e-6 H2G_MICROBATCH_SIZE=4 H2G_TENSOR_PARALLEL_SIZE=1
export H2G_VAL_DATA_DIR="$ST/validation/historical_148480_v1" H2G_ONLINE_VALIDATION_RECEIPT="$ST/receipts/historical_online_validation_148480_canonical_v2.json"
export H2G_EXTRA_VALID_SETS="hplt openarchives greek_phd english de ru zh code old_greek" H2G_NEW_GREEK_VALID_SETS="greek_phd hplt openarchives"
export H2G_MEGATRON_DIR="$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2" H2G_MEGATRON_RECEIPT="$ST/receipts/training_megatron_runtime_helpers_v2.json"
export H2G_TRAINING_RUN_PERMIT="$ST/receipts/training_run_permit_1p5b_v15.json" H2G_AUTHORIZATION_GATE="$ST/receipts/launch_gate_pre_extension.json" H2G_STATIC_PREFLIGHT="$ST/receipts/canonical_static_preflight_v15/1p5b_s4.json"
export H2G_ATTEMPT_TAG=1p5b_s4_3146990_v15 H2G_SRUN_WRAPPER_DIR="$ST/control/retries/srun_node_gpu_visibility_v2" H2G_TIME_LIMIT=01:30:00
exec uenv run pytorch/v2.9.1:v2 --view=default -- bash "$ST/control/retries/train_hard_h_to_g_segment_runtime_overlay_v1.sbatch"
