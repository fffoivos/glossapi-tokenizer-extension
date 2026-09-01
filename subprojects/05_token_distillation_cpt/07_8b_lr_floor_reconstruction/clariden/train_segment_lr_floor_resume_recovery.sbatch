#!/usr/bin/env bash
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=08:00:00
#SBATCH --signal=SIGUSR2@600
#SBATCH --no-requeue

set -euo pipefail
: "${LR13_CODE_ROOT:?set LR13_CODE_ROOT}"
: "${LR13_ASSETS_RECEIPT:?set LR13_ASSETS_RECEIPT}"
: "${LR13_START_ITERATION:?set LR13_START_ITERATION}"
: "${LR13_END_ITERATION:?set LR13_END_ITERATION}"
: "${LR13_PHASE:?set LR13_PHASE}"
: "${LR13_LR_FLOOR_PERCENT:?set LR13_LR_FLOOR_PERCENT}"
: "${LR13_LOAD_CHECKPOINT:?set LR13_LOAD_CHECKPOINT}"
: "${LR13_SOURCE_REPO:?set LR13_SOURCE_REPO}"
: "${LR13_OUTPUT_DIR:?set LR13_OUTPUT_DIR}"
: "${LR13_RECOVERY_BUNDLE:?set LR13_RECOVERY_BUNDLE}"

preflight=(python3 "$LR13_CODE_ROOT/clariden/preflight_segment.py"
  --assets "$LR13_ASSETS_RECEIPT"
  --start "$LR13_START_ITERATION" --end "$LR13_END_ITERATION"
  --phase "$LR13_PHASE" --lr-floor-percent "$LR13_LR_FLOOR_PERCENT"
  --load-checkpoint "$LR13_LOAD_CHECKPOINT")
if [[ -n "${LR13_RESUME_RECEIPT:-}" ]]; then
  preflight+=(--resume-receipt "$LR13_RESUME_RECEIPT")
fi
uenv run pytorch/v2.9.1:v2 --view=default -- "${preflight[@]}"

export ARM=td
export INIT_CKPT="$LR13_LOAD_CHECKPOINT"
export OUTPUT_DIR="$LR13_OUTPUT_DIR"
export TRAIN_CONFIG_OVERRIDE="$LR13_RECOVERY_BUNDLE/lr_floor_config_recovery.env"
export SCRIPT_DIR_OVERRIDE="$LR13_SOURCE_REPO/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training"
export RESUME_TRAINING=1
export EXIT_INTERVAL="$LR13_END_ITERATION"
export DISABLE_SAVE=0
export SAVE_INTERVAL="$LR13_SAVE_INTERVAL"
export ACCOUNT=a0140 PARTITION=normal NODES=16 GPUS_PER_NODE=4 LAUNCH_MODE=torchrun
export TIME_LIMIT="${LR13_TIME_LIMIT:-08:00:00}"
export MEGATRON_LM_SWISSAI_DIR
exec bash "$SCRIPT_DIR_OVERRIDE/bakeoff_train.sbatch"
