#!/usr/bin/env bash
# Per-arm eval: takes one arm's checkpoint directory and runs the full eval suite.
# Use during the bakeoff (per v0.7 §6.1: every 500 M tokens for downstream;
# windowed average across last 3-5 checkpoints in 80-100% budget range for selection).
#
# Thin wrapper around run_eval.sbatch.
#
# Usage:
#   bash run_bakeoff_arm_eval.sh <arm-checkpoint-dir> [task-group]
#
# Examples:
#   bash run_bakeoff_arm_eval.sh /iopsstor/.../init_checkpoints/retok
#   bash run_bakeoff_arm_eval.sh /capstor/.../runs/retok_bakeoff/checkpoint-1500 greek_only

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <arm-checkpoint-dir> [task-group=full]" >&2
    exit 2
fi

CKPT_DIR="$1"
TASK_GROUP="${2:-full}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: checkpoint dir does not exist: $CKPT_DIR" >&2
    exit 2
fi

OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
ARM_NAME=$(basename "$(dirname "$CKPT_DIR")")_$(basename "$CKPT_DIR")
RUN_NAME="bakeoff_${ARM_NAME}_$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Submitting bakeoff arm eval"
echo "  MODEL_PATH=$CKPT_DIR"
echo "  TASK_GROUP=$TASK_GROUP"
echo "  OUTPUT_DIR=$OUTPUT_DIR"

sbatch \
    --export=ALL,MODEL_PATH="$CKPT_DIR",OUTPUT_DIR="$OUTPUT_DIR",TASK_GROUP="$TASK_GROUP" \
    --job-name="eval_$ARM_NAME" \
    "$(dirname "$0")/run_eval.sbatch"

echo
echo "When done, run:"
echo "  python3 compute_bootstrap_cis.py $OUTPUT_DIR/samples_*.jsonl > $OUTPUT_DIR/bootstrap_cis.json"
