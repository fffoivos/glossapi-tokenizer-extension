#!/usr/bin/env bash
# Submit the decision-useful 2B Token Distillation layer11 arm.
#
# Clariden normal has a 12h walltime limit. The one-node smoke estimates this
# arm at about 17.5h, so the default submission is a two-job chain: an initial
# job from the selected TD checkpoint, followed by a resume job from the run's
# own checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_train_config_common.env"

TD_INIT_CKPT="${TD_INIT_CKPT:-/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
RUN_TAG="${RUN_TAG:-td_full25_layer11_2b_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="$OUT_ROOT/$RUN_TAG"
RUN_TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
CHAIN_RESUME="${CHAIN_RESUME:-1}"
RUN_SAVE_INTERVAL="${SAVE_INTERVAL:-65}"
RUN_EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"
RUN_EXIT_INTERVAL="${EXIT_INTERVAL:-}"
RUN_DISABLE_SAVE="${DISABLE_SAVE:-0}"
SBATCH_NTASKS_PER_NODE="$GPUS_PER_NODE"
if [ "$LAUNCH_MODE" = "torchrun" ]; then
    SBATCH_NTASKS_PER_NODE="1"
fi

if [ "$CHAIN_RESUME" != "0" ] && [ "$CHAIN_RESUME" != "1" ]; then
    echo "ERROR: CHAIN_RESUME=$CHAIN_RESUME not recognized (expected 0|1)" >&2
    exit 2
fi
if [ "$RUN_DISABLE_SAVE" != "0" ]; then
    echo "ERROR: production 2B chain requires DISABLE_SAVE=0" >&2
    exit 2
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

echo "=== submit_td_layer11_2b_chain.sh ==="
echo "TD_INIT_CKPT: $TD_INIT_CKPT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "EXT_DATA_PREFIX: $EXT_DATA_PREFIX"
echo "ACCOUNT: $ACCOUNT"
echo "PARTITION: $PARTITION"
echo "NODES: $NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "LAUNCH_MODE: $LAUNCH_MODE"
echo "TIME_LIMIT: $RUN_TIME_LIMIT"
echo "SAVE_INTERVAL: $RUN_SAVE_INTERVAL"
echo "EVAL_INTERVAL: $RUN_EVAL_INTERVAL"
echo "EXIT_INTERVAL: ${RUN_EXIT_INTERVAL:-<none>}"
echo "CHAIN_RESUME: $CHAIN_RESUME"

initial_job_id=$(sbatch --parsable \
    --job-name="td_l11_2b" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --nodes="$NODES" \
    --ntasks-per-node="$SBATCH_NTASKS_PER_NODE" \
    --gpus-per-node="$GPUS_PER_NODE" \
    --gres="gpu:$GPUS_PER_NODE" \
    --time="$RUN_TIME_LIMIT" \
    --export=ALL,ARM=retok,INIT_CKPT="$TD_INIT_CKPT",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$RUN_TIME_LIMIT",SAVE_INTERVAL="$RUN_SAVE_INTERVAL",EVAL_INTERVAL="$RUN_EVAL_INTERVAL",EXIT_INTERVAL="$RUN_EXIT_INTERVAL",DISABLE_SAVE=0,RESUME_TRAINING=0 \
    "$SCRIPT_DIR/bakeoff_train.sbatch")
echo "initial job: $initial_job_id"

if [ "$CHAIN_RESUME" = "1" ]; then
    resume_job_id=$(sbatch --parsable \
        --dependency="afterany:$initial_job_id" \
        --job-name="td_l11_2b_resume" \
        --account="$ACCOUNT" \
        --partition="$PARTITION" \
        --nodes="$NODES" \
        --ntasks-per-node="$SBATCH_NTASKS_PER_NODE" \
        --gpus-per-node="$GPUS_PER_NODE" \
        --gres="gpu:$GPUS_PER_NODE" \
        --time="$RUN_TIME_LIMIT" \
        --export=ALL,ARM=retok,INIT_CKPT="$OUTPUT_DIR/checkpoints",OUTPUT_DIR="$OUTPUT_DIR",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",ACCOUNT="$ACCOUNT",PARTITION="$PARTITION",NODES="$NODES",GPUS_PER_NODE="$GPUS_PER_NODE",LAUNCH_MODE="$LAUNCH_MODE",TIME_LIMIT="$RUN_TIME_LIMIT",SAVE_INTERVAL="$RUN_SAVE_INTERVAL",EVAL_INTERVAL="$RUN_EVAL_INTERVAL",EXIT_INTERVAL="$RUN_EXIT_INTERVAL",DISABLE_SAVE=0,RESUME_TRAINING=1 \
        "$SCRIPT_DIR/bakeoff_train.sbatch")
    echo "resume job:  $resume_job_id (afterany:$initial_job_id)"
fi
