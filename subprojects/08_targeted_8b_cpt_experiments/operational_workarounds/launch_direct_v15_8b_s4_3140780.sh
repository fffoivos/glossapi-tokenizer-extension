#!/usr/bin/env bash
set -euo pipefail
JOB=3140780
ST=/capstor/scratch/cscs/fffoivos/cpt_runs/hard_h2g_matched/20260814T201715Z-r2-v14
CODE=/iopsstor/scratch/cscs/fffoivos/orchestration/targeted-8b-cpt/20260822T014000Z-hard-h2g-ledger-invariant-v15
SUB="$CODE/subprojects/08_targeted_8b_cpt_experiments"
RUN8="$ST/runs/efficiency_bound_proven_8b_v112_recovery_codebinding_v3_staticready"
WRAPPER="$ST/control/retries/srun_node_gpu_visibility_v2"
record=$(scontrol show job -o "$JOB")
[[ "$record" == *"JobState=RUNNING"* && "$record" == *"Partition=normal"* && "$record" == *"NumNodes=16"* ]] || exit 2
nodelist=$(squeue -h -j "$JOB" -o '%N')
end_epoch=$(date -d "$(squeue -h -j "$JOB" -o '%e')" +%s)
(( end_epoch - $(date +%s) >= 5400 )) || { echo "insufficient 8B allocation time" >&2; exit 2; }
export SLURM_JOB_ID="$JOB" SLURM_JOBID="$JOB" SLURM_JOB_ACCOUNT=a0140 SLURM_JOB_PARTITION=normal
export SLURM_JOB_NODELIST="$nodelist" SLURM_NODELIST="$nodelist" SLURM_JOB_NUM_NODES=16 SLURM_NNODES=16
export SLURM_NTASKS=16 SLURM_NPROCS=16 SLURM_TASKS_PER_NODE='1(x16)' SLURM_CPUS_PER_TASK=288
export SLURM_GPUS_ON_NODE=4 SLURM_GPUS_PER_NODE=4
export H2G_CODE_ROOT="$CODE" H2G_CODE_RECEIPT="$CODE.receipt.json" H2G_STAGE_ROOT="$ST"
export H2G_MODEL_SCALE=8b H2G_PHASE=3 H2G_START_UPDATE=3218 H2G_EXIT_UPDATE=3456
export H2G_LOAD_CHECKPOINT="$RUN8/segments/s3/attempts/attempt_000003/payload/checkpoints"
export H2G_CHECKPOINT_PERMIT="$RUN8/segments/s3/attempts/attempt_000003/checkpoint_permit.ca9dd64_postprocess_fix.json"
export H2G_SOURCE_PHASE_CACHE_RECEIPT="$ST/receipts/phase_2_blend_cache_runtime_exact_v103_postprocess_recovery.json"
export H2G_PHASE_DATA_PATH_SPEC="$ST/data/phases/phase3/phase_data_path.json"
export H2G_PHASE_DATA_PATH="1.0 $ST/megatron/phase3_openarchives_ext_text_document 0.253164557 $ST/megatron/phase3_foreign_ext_text_document 0.012658228 $ST/megatron/phase3_old_greek_ext_text_document"
export H2G_PHASE_CACHE_RECEIPT="$ST/receipts/phase_3_blend_cache.json" H2G_PHASE_CACHE_ROOT="$ST/data/phases/phase3/cache"
export H2G_PHASE_CACHE_TREE_SHA256=6575b7e478a1db225facb2d2c2ea6edc62e7dc7cfb0a6a5d9d7b56698f1d2260
export H2G_RUNTIME_PHASE_CACHE_ROOT="$ST/cache_overlays/phase3_8b_historical_validation_v14"
export H2G_CACHE_OVERLAY_RECEIPT="$ST/receipts/phase_3_overlay_8b_historical_validation_v14.json"
export H2G_PRODUCER_COMPATIBILITY="$ST/receipts/producer_bundle_compatibility_extension_gate_v15_phase3_ledger.json"
export H2G_SEGMENT_OUTPUT_ROOT="$ST/runs/direct_v15_8b_s4_3218_3456_3140780"
export H2G_PEAK_LR=5.5e-5 H2G_FLOOR_LR=5.5e-6 H2G_MICROBATCH_SIZE=2 H2G_TENSOR_PARALLEL_SIZE=2
export H2G_VAL_DATA_DIR="$ST/validation/historical_148480_v1" H2G_ONLINE_VALIDATION_RECEIPT="$ST/receipts/historical_online_validation_148480_canonical_v2.json"
export H2G_EXTRA_VALID_SETS="hplt openarchives greek_phd english de ru zh code old_greek" H2G_NEW_GREEK_VALID_SETS="greek_phd hplt openarchives"
export H2G_MEGATRON_DIR="$ST/tools/megatron_training_c92402e_extra_valid_helpers_v2" H2G_MEGATRON_RECEIPT="$ST/receipts/training_megatron_runtime_helpers_v2.json"
export H2G_TRAINING_RUN_PERMIT="$ST/receipts/training_run_permit_8b_v15.json" H2G_AUTHORIZATION_GATE="$ST/receipts/launch_gate_pre_extension.json"
export H2G_STATIC_PREFLIGHT="$ST/receipts/canonical_static_preflight_v15/8b_s4.json" H2G_ATTEMPT_TAG=8b_s4_3140780_v15
export H2G_SRUN_WRAPPER_DIR="$WRAPPER" H2G_TIME_LIMIT=04:00:00
exec uenv run pytorch/v2.9.1:v2 --view=default -- bash "$ST/control/retries/train_hard_h_to_g_segment_runtime_overlay_v1.sbatch"
