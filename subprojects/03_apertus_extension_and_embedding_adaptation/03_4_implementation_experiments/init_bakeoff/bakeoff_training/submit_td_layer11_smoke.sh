#!/usr/bin/env bash
# Submit a bounded one-arm smoke that loads the selected Token Distillation
# layer11 R17-patched Megatron checkpoint and exits after a few train iterations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USER_TIME_LIMIT="${TIME_LIMIT:-}"
USER_SAVE_INTERVAL="${SAVE_INTERVAL:-}"
USER_EVAL_INTERVAL="${EVAL_INTERVAL:-}"
USER_EXIT_INTERVAL="${EXIT_INTERVAL:-}"

source "$SCRIPT_DIR/_train_config_common.env"

TD_INIT_CKPT="${TD_INIT_CKPT:-/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
RUN_TAG="${RUN_TAG:-smoke_td_full25_layer11_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$OUT_ROOT/$RUN_TAG"
SMOKE_TIME_LIMIT="${USER_TIME_LIMIT:-01:00:00}"
SMOKE_SAVE_INTERVAL="${USER_SAVE_INTERVAL:-999999}"
SMOKE_EVAL_INTERVAL="${USER_EVAL_INTERVAL:-999999}"
SMOKE_EXIT_INTERVAL="${USER_EXIT_INTERVAL:-5}"
SMOKE_DISABLE_SAVE="${DISABLE_SAVE:-1}"
SBATCH_NTASKS_PER_NODE="$GPUS_PER_NODE"
if [ "$LAUNCH_MODE" = "torchrun" ]; then
    SBATCH_NTASKS_PER_NODE="1"
fi

if [ ! -d "$TD_INIT_CKPT/release" ]; then
    echo "ERROR: missing TD patched checkpoint: $TD_INIT_CKPT/release" >&2
    exit 2
fi
if [ ! -f "${EXT_DATA_PREFIX}.bin" ] || [ ! -f "${EXT_DATA_PREFIX}.idx" ]; then
    echo "ERROR: missing extended-tokenized Megatron data at ${EXT_DATA_PREFIX}{.bin,.idx}" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "=== submit_td_layer11_smoke.sh ==="
echo "TD_INIT_CKPT: $TD_INIT_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EXT_DATA_PREFIX: $EXT_DATA_PREFIX"
echo "ACCOUNT: $ACCOUNT"
echo "PARTITION: $PARTITION"
echo "NODES: $NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "LAUNCH_MODE: $LAUNCH_MODE"
echo "TIME_LIMIT: $SMOKE_TIME_LIMIT"
echo "SAVE_INTERVAL: $SMOKE_SAVE_INTERVAL"
echo "EVAL_INTERVAL: $SMOKE_EVAL_INTERVAL"
echo "EXIT_INTERVAL: $SMOKE_EXIT_INTERVAL"
echo "DISABLE_SAVE: $SMOKE_DISABLE_SAVE"

sbatch \
    --job-name="bakeoff_smoke_td_l11" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes="$NODES" \
    --ntasks-per-node="$SBATCH_NTASKS_PER_NODE" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --gres="gpu:$GPUS_PER_NODE" \
    --time="$SMOKE_TIME_LIMIT" \
    --export=ALL,ARM=retok,INIT_CKPT="$TD_INIT_CKPT",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$SMOKE_TIME_LIMIT",SAVE_INTERVAL="$SMOKE_SAVE_INTERVAL",EVAL_INTERVAL="$SMOKE_EVAL_INTERVAL",EXIT_INTERVAL="$SMOKE_EXIT_INTERVAL",DISABLE_SAVE="$SMOKE_DISABLE_SAVE" \
    "$SCRIPT_DIR/bakeoff_train.sbatch"
