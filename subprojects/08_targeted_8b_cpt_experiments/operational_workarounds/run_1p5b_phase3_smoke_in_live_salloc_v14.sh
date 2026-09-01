#!/usr/bin/env bash
set -euo pipefail

# Experiment-local adapter for the already-held two-node 1.5B allocation.
# It changes only the immutable code/gate binding from V12 to the joint V14
# extension authority; all checkpoint, data, optimizer, LR and GPU geometry
# bindings remain identical to the previously staged launcher.

ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
RUN="$ST/runs/h2g_1p5b_2node_v1_retry14_direct_salloc_uenv_20260821"
ATTEMPT="$RUN/segments/s3/attempts/attempt_000002"
CODE=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260821T183000Z-hard-h2g-phase3-lineage-v14
WRAPPER="$ST/control/retries/srun_node_gpu_visibility_v2"

[[ "${SLURM_JOB_ID:-}" == 3141832 ]] || { echo "expected live allocation 3141832" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == normal ]] || { echo "normal allocation required" >&2; exit 2; }
[[ "${SLURM_JOB_NUM_NODES:-0}" == 2 ]] || { echo "exact two-node allocation required" >&2; exit 2; }
[[ "${SLURM_NTASKS:-0}" == 8 ]] || { echo "exact eight-task allocation required" >&2; exit 2; }
[[ -x "$WRAPPER/srun" ]] || { echo "audited srun wrapper missing" >&2; exit 2; }

tracker="$ATTEMPT/payload/checkpoints/latest_checkpointed_iteration.txt"
permit="$ATTEMPT/checkpoint_permit.json"
[[ -f "$tracker" && "$(<"$tracker")" == 3218 ]] || { echo "1.5B phase-2 endpoint 3218 is not frozen" >&2; exit 2; }
[[ -f "$permit" ]] || { echo "1.5B phase-2 checkpoint permit missing" >&2; exit 2; }

export PATH="$WRAPPER:$PATH"
export H2G_CODE_ROOT="$CODE"
export H2G_CODE_RECEIPT="$CODE.receipt.json"
export H2G_STAGE_ROOT="$ST"
export H2G_MODEL_SCALE=1p5b
export H2G_START_UPDATE=3218
export H2G_LOAD_CHECKPOINT="$ATTEMPT/payload/checkpoints"
export H2G_CHECKPOINT_PERMIT="$permit"
export H2G_SOURCE_PHASE_CACHE_RECEIPT="$ST/receipts/phase_2_blend_cache_1p5b_val148_v116_v10_postprocess_rebind.json"
export H2G_PHASE_DATA_PATH_SPEC="$ST/data/phases/phase3/phase_data_path.json"
export H2G_PHASE_DATA_PATH="1.0 $ST/megatron/phase3_openarchives_ext_text_document 0.253164557 $ST/megatron/phase3_foreign_ext_text_document 0.012658228 $ST/megatron/phase3_old_greek_ext_text_document"
export H2G_PHASE_CACHE_RECEIPT="$ST/receipts/phase_3_blend_cache.json"
export H2G_PHASE_CACHE_ROOT="$ST/data/phases/phase3/cache"
export H2G_PHASE_CACHE_TREE_SHA256=6575b7e478a1db225facb2d2c2ea6edc62e7dc7cfb0a6a5d9d7b56698f1d2260
export H2G_SMOKE_OUTPUT_ROOT="$ST/runs/phase3_resume_smoke_1p5b_3218_v14_direct_salloc_3141832_v8"
export H2G_PEAK_LR=5.5e-5
export H2G_FLOOR_LR=5.5e-6
export H2G_MICROBATCH_SIZE=4
export H2G_TENSOR_PARALLEL_SIZE=1
export H2G_VAL_DATA_DIR="$ST/validation/historical_148480_v1"
export H2G_ONLINE_VALIDATION_RECEIPT="$ST/receipts/historical_online_validation_148480_canonical_v2.json"
export H2G_EXTRA_VALID_SETS="hplt openarchives greek_phd english de ru zh code old_greek"
export H2G_NEW_GREEK_VALID_SETS="greek_phd hplt openarchives"
export H2G_MEGATRON_DIR="$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2"
export H2G_MEGATRON_RECEIPT="$ST/receipts/training_megatron_runtime_helpers_v2.json"
export H2G_TRAINING_RUN_PERMIT="$ST/receipts/training_run_permit_1p5b_tp1_dp8_2node_v14_phase3.json"
export H2G_AUTHORIZATION_GATE="$ST/receipts/launch_gate_1p5b_pre_main_v14_phase3_smoke.json"
export H2G_STATIC_PREFLIGHT="$ST/receipts/canonical_static_preflight_phase3_v14/1p5b_smoke_3218.json"
export H2G_TIME_LIMIT=00:20:00
export H2G_SMOKE_ATTEMPT_TAG=3141832_gpu_v9
export H2G_SRUN_WRAPPER_DIR="$WRAPPER"
export H2G_RUNTIME_COMPAT_DIR="$CODE/subprojects/08_targeted_8b_cpt_experiments/runtime_compat"
export H2G_RUNTIME_PHASE_CACHE_ROOT="$ST/cache_overlays/phase3_8b_historical_validation_v14"
export H2G_CACHE_OVERLAY_RECEIPT="$ST/receipts/phase_3_overlay_8b_historical_validation_v14.json"
export H2G_PRODUCER_COMPATIBILITY="$ST/receipts/producer_bundle_compatibility_extension_gate_v14_phase3.json"

exec uenv run pytorch/v2.9.1:v2 --view=default -- \
  bash "$ST/control/retries/run_phase3_resume_smoke_retryable_v7.sbatch"
