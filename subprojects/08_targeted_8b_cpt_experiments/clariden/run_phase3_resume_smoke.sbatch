#!/usr/bin/env bash
#SBATCH --job-name=h2g-p3-resume-smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_MODEL_SCALE H2G_START_UPDATE H2G_LOAD_CHECKPOINT H2G_CHECKPOINT_PERMIT H2G_SOURCE_PHASE_CACHE_RECEIPT H2G_PHASE_DATA_PATH_SPEC H2G_PHASE_DATA_PATH H2G_PHASE_CACHE_RECEIPT H2G_PHASE_CACHE_ROOT H2G_PHASE_CACHE_TREE_SHA256 H2G_SMOKE_OUTPUT_ROOT H2G_PEAK_LR H2G_FLOOR_LR H2G_MICROBATCH_SIZE H2G_TENSOR_PARALLEL_SIZE H2G_VAL_DATA_DIR H2G_ONLINE_VALIDATION_RECEIPT H2G_EXTRA_VALID_SETS H2G_NEW_GREEK_VALID_SETS H2G_MEGATRON_DIR H2G_MEGATRON_RECEIPT H2G_TRAINING_RUN_PERMIT H2G_AUTHORIZATION_GATE; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == normal ]] || { echo "Phase-3 resume smoke requires normal" >&2; exit 2; }
[[ "$H2G_START_UPDATE" == 3218 || "$H2G_START_UPDATE" == 3456 ]] || { echo "invalid Phase-3 smoke start" >&2; exit 2; }
[[ ! -e "$H2G_SMOKE_OUTPUT_ROOT" ]] || { echo "immutable smoke output exists" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
end_update=$((H2G_START_UPDATE + 1))
preflight="$H2G_STAGE_ROOT/receipts/phase3_smoke_preflight_${H2G_MODEL_SCALE}_${H2G_START_UPDATE}_${SLURM_JOB_ID}.json"
/usr/bin/python3.11 "$subproject/scripts/preflight_train_segment.py" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --scale "$H2G_MODEL_SCALE" --phase 3 --start-update "$H2G_START_UPDATE" --exit-update "$end_update" \
  --load-checkpoint "$H2G_LOAD_CHECKPOINT" --checkpoint-permit "$H2G_CHECKPOINT_PERMIT" \
  --source-phase-cache-receipt "$H2G_SOURCE_PHASE_CACHE_RECEIPT" \
  --phase-data-path-spec "$H2G_PHASE_DATA_PATH_SPEC" --phase-data-path "$H2G_PHASE_DATA_PATH" \
  --phase-cache-receipt "$H2G_PHASE_CACHE_RECEIPT" --phase-cache-root "$H2G_PHASE_CACHE_ROOT" \
  --phase-cache-tree-sha256 "$H2G_PHASE_CACHE_TREE_SHA256" \
  --megatron-root "$H2G_MEGATRON_DIR" --megatron-receipt "$H2G_MEGATRON_RECEIPT" \
  --validation-root "$H2G_VAL_DATA_DIR" --validation-receipt "$H2G_ONLINE_VALIDATION_RECEIPT" \
  --extra-valid-sets "$H2G_EXTRA_VALID_SETS" --new-greek-valid-sets "$H2G_NEW_GREEK_VALID_SETS" \
  --training-run-permit "$H2G_TRAINING_RUN_PERMIT" --peak-lr "$H2G_PEAK_LR" --floor-lr "$H2G_FLOOR_LR" \
  --authorization-gate "$H2G_AUTHORIZATION_GATE" \
  --nodes "$SLURM_NNODES" --tensor-parallel "$H2G_TENSOR_PARALLEL_SIZE" --microbatch "$H2G_MICROBATCH_SIZE" \
  --one-update-resume-smoke --output "$preflight"

export H2G_PHASE=3 H2G_EXIT_UPDATE="$end_update"
export TRAIN_CONFIG_OVERRIDE="$subproject/clariden/hard_h_to_g_training.env"
export SCRIPT_DIR_OVERRIDE="$H2G_CODE_ROOT/frozen_training_tools/bakeoff_training"
export ARM=td INIT_CKPT="$H2G_LOAD_CHECKPOINT" OUTPUT_DIR="$H2G_SMOKE_OUTPUT_ROOT"
export ACCOUNT=a0140 PARTITION=normal NODES="$SLURM_NNODES" GPUS_PER_NODE=4 LAUNCH_MODE=torchrun
export H2G_SAVE_INTERVAL_OVERRIDE=1 H2G_EVAL_INTERVAL_OVERRIDE=1000000 FULL_CPT_LAUNCH_VERIFY=0
export TIME_LIMIT="${H2G_TIME_LIMIT:-01:30:00}"
mkdir -p "$(dirname "$H2G_SMOKE_OUTPUT_ROOT")"
started_epoch=$(date +%s)
bash "$SCRIPT_DIR_OVERRIDE/bakeoff_train.sbatch" > "$H2G_SMOKE_OUTPUT_ROOT.driver.partial" 2>&1
mv "$H2G_SMOKE_OUTPUT_ROOT.driver.partial" "$H2G_SMOKE_OUTPUT_ROOT/driver.out"
completed_epoch=$(date +%s)
printf '%s\n' "$((completed_epoch - started_epoch))" > "$H2G_SMOKE_OUTPUT_ROOT/wall_seconds.txt"
cp "$preflight" "$H2G_SMOKE_OUTPUT_ROOT/preflight.json"
chmod -R a-w "$H2G_SMOKE_OUTPUT_ROOT"
