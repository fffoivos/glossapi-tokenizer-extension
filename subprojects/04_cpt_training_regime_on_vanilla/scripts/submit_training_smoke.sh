#!/usr/bin/env bash
# Submit the short Vanilla Goldfish training smoke after dataset smoke passes.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/04_cpt_training_regime_on_vanilla"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
TRAIN_SCRIPT="$LEGACY_INIT_DIR/bakeoff_training/bakeoff_train.sbatch"
TRAIN_CONFIG="$SUBPROJECT_DIR/scripts/train_config_04_vanilla.env"
INIT_CKPT="${INIT_CKPT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt}"
RUN_TAG="${RUN_TAG:-04_vanilla_goldfish_smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/$RUN_TAG}"
TRAIN_PARTITION="${TRAIN_PARTITION:-normal}"
TRAIN_TIME="${TRAIN_TIME:-02:30:00}"

if [ -z "${DATA_PREFIX:-}" ]; then
  if [ -f "$RUN_ROOT/latest_dataset/dataset_paths.env" ]; then
    # shellcheck disable=SC1090
    source "$RUN_ROOT/latest_dataset/dataset_paths.env"
  else
    echo "ERROR: set DATA_PREFIX or run hplt_b1_dataset_smoke.sbatch first." >&2
    exit 2
  fi
fi

mkdir -p "$OUTPUT_DIR"
test -f "$TRAIN_SCRIPT"
test -f "$TRAIN_CONFIG"
test -d "$INIT_CKPT/release"
test -f "$DATA_PREFIX.bin"
test -f "$DATA_PREFIX.idx"

echo "Submitting 04 Vanilla Goldfish training smoke"
echo "  TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "  TRAIN_CONFIG=$TRAIN_CONFIG"
echo "  INIT_CKPT=$INIT_CKPT"
echo "  DATA_PREFIX=$DATA_PREFIX"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "  TRAIN_PARTITION=$TRAIN_PARTITION"
echo "  TRAIN_TIME=$TRAIN_TIME"
echo

job_id="$(sbatch --parsable \
  --account=a0140 \
  --partition="$TRAIN_PARTITION" \
  --nodes=1 \
  --ntasks-per-node=4 \
  --gpus-per-node=4 \
  --gres=gpu:4 \
  --cpus-per-task=36 \
  --time="$TRAIN_TIME" \
  --job-name=04van_gold_smoke \
  --output="$RUN_ROOT/%x-%j.out" \
  --error="$RUN_ROOT/%x-%j.err" \
  --export=ALL,ARM=vanilla,INIT_CKPT="$INIT_CKPT",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$LEGACY_INIT_DIR/bakeoff_training",TRAIN_CONFIG_OVERRIDE="$TRAIN_CONFIG",BASE_DATA_PREFIX="$DATA_PREFIX",TRAIN_TOKENS=5000000000,SAVE_INTERVAL=2,EXIT_INTERVAL=2,LOSS_OBJECTIVE=goldfish,LR_SCHEDULE_STYLE=constant,LR_WARMUP_INIT=1.1e-6,ADEMA_BETA3_WARMUP_STEPS=287,ADEMA_ALPHA_WARMUP_STEPS=287,DATA_SEED=20260528 \
  "$TRAIN_SCRIPT")"

echo "$job_id" > "$OUTPUT_DIR/training_smoke_job.id"
cat > "$OUTPUT_DIR/training_smoke_paths.env" <<PATHS
TRAIN_JOB_ID="$job_id"
OUTPUT_DIR="$OUTPUT_DIR"
DATA_PREFIX="$DATA_PREFIX"
INIT_CKPT="$INIT_CKPT"
TRAIN_CONFIG="$TRAIN_CONFIG"
PATHS

echo "Submitted training smoke job: $job_id"
echo "Monitor:"
echo "  squeue -j $job_id -o '%.18i %.32j %.2t %.10M %.10l %D %R'"
