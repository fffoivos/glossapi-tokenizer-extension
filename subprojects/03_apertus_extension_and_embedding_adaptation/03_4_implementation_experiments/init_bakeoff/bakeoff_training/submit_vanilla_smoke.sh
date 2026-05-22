#!/usr/bin/env bash
# Submit a one-arm smoke that loads the patched vanilla checkpoint, exercises
# first iterations, writes an early checkpoint, and exits cleanly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USER_TIME_LIMIT="${TIME_LIMIT:-}"
USER_SAVE_INTERVAL="${SAVE_INTERVAL:-}"
USER_EVAL_INTERVAL="${EVAL_INTERVAL:-}"
USER_EXIT_INTERVAL="${EXIT_INTERVAL:-}"

source "$SCRIPT_DIR/_train_config_common.env"

INIT_CKPT_ROOT="${INIT_CKPT_ROOT:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480}"
INIT_CKPT_SUBDIR="${INIT_CKPT_SUBDIR:-megatron_tp2_r17patched}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
RUN_TAG="${RUN_TAG:-smoke_vanilla_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$OUT_ROOT/$RUN_TAG"
INIT_CKPT="$INIT_CKPT_ROOT/vanilla/$INIT_CKPT_SUBDIR"
SMOKE_TIME_LIMIT="${USER_TIME_LIMIT:-01:00:00}"
SMOKE_SAVE_INTERVAL="${USER_SAVE_INTERVAL:-5}"
SMOKE_EVAL_INTERVAL="${USER_EVAL_INTERVAL:-999999}"
SMOKE_EXIT_INTERVAL="${USER_EXIT_INTERVAL:-10}"
SBATCH_NTASKS_PER_NODE="$GPUS_PER_NODE"
if [ "$LAUNCH_MODE" = "torchrun" ]; then
    SBATCH_NTASKS_PER_NODE="1"
fi

if [ ! -d "$INIT_CKPT/release" ]; then
    echo "ERROR: missing patched vanilla checkpoint: $INIT_CKPT/release" >&2
    exit 2
fi
if [ ! -f "${BASE_DATA_PREFIX}.bin" ] || [ ! -f "${BASE_DATA_PREFIX}.idx" ]; then
    echo "ERROR: missing base-tokenized Megatron data at ${BASE_DATA_PREFIX}{.bin,.idx}" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "=== submit_vanilla_smoke.sh ==="
echo "INIT_CKPT: $INIT_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "BASE_DATA_PREFIX: $BASE_DATA_PREFIX"
echo "ACCOUNT: $ACCOUNT"
echo "PARTITION: $PARTITION"
echo "NODES: $NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "LAUNCH_MODE: $LAUNCH_MODE"
echo "TIME_LIMIT: $SMOKE_TIME_LIMIT"
echo "SAVE_INTERVAL: $SMOKE_SAVE_INTERVAL"
echo "EVAL_INTERVAL: $SMOKE_EVAL_INTERVAL"
echo "EXIT_INTERVAL: $SMOKE_EXIT_INTERVAL"

sbatch \
    --job-name="bakeoff_smoke_vanilla" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes="$NODES" \
    --ntasks-per-node="$SBATCH_NTASKS_PER_NODE" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --gres="gpu:$GPUS_PER_NODE" \
    --time="$SMOKE_TIME_LIMIT" \
    --export=ALL,ARM=vanilla,INIT_CKPT="$INIT_CKPT",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$SMOKE_TIME_LIMIT",SAVE_INTERVAL="$SMOKE_SAVE_INTERVAL",EVAL_INTERVAL="$SMOKE_EVAL_INTERVAL",EXIT_INTERVAL="$SMOKE_EXIT_INTERVAL" \
    "$SCRIPT_DIR/bakeoff_train.sbatch"
