#!/usr/bin/env bash
# Submit conversion + lm-eval for a saved bakeoff Megatron checkpoint.
#
# Usage:
#   RUN_TAG=<run-tag> bash submit_bakeoff_checkpoint_eval.sh <arm> <iter> [task-group]
#
# Examples:
#   RUN_TAG=bakeoff_1node_chain_20260522_005620 bash submit_bakeoff_checkpoint_eval.sh vanilla 65 greek_only
#   RUN_TAG=bakeoff_1node_chain_20260522_005620 EVAL_JSONL=/path/heldout.jsonl SUBMIT_INTRINSIC=1 bash submit_bakeoff_checkpoint_eval.sh retok 130 full

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: RUN_TAG=<run-tag> $0 <arm> <iter> [task-group=full]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARM="$1"
ITER="$2"
TASK_GROUP="${3:-full}"
RUN_TAG="${RUN_TAG:?RUN_TAG is required, e.g. bakeoff_1node_chain_20260522_005620}"

case "$ARM" in
    vanilla|retok|centroid) ;;
    *) echo "ERROR: arm must be vanilla|retok|centroid, got: $ARM" >&2; exit 2 ;;
esac
case "$TASK_GROUP" in
    full|retention_only|greek_only|safety_only) ;;
    *) echo "ERROR: task-group must be full|retention_only|greek_only|safety_only, got: $TASK_GROUP" >&2; exit 2 ;;
esac
if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: iter must be an integer, got: $ITER" >&2
    exit 2
fi

ITER_PAD="$(printf "%07d" "$ITER")"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
MEGATRON_CKPT_ROOT="${MEGATRON_CKPT_ROOT:-$RUN_ROOT/${RUN_TAG}_${ARM}/checkpoints}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
ARM_EVAL_ROOT="$OUT_ROOT/${RUN_TAG}_${ARM}"
HF_OUT_DIR="${HF_OUT_DIR:-$ARM_EVAL_ROOT/iter_${ITER_PAD}_hf}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ARM_EVAL_ROOT/iter_${ITER_PAD}_${TASK_GROUP}}"

if [ ! -d "$MEGATRON_CKPT_ROOT/iter_$ITER_PAD" ]; then
    echo "ERROR: missing checkpoint: $MEGATRON_CKPT_ROOT/iter_$ITER_PAD" >&2
    exit 3
fi

mkdir -p "$ARM_EVAL_ROOT"

echo "Submitting bakeoff checkpoint eval"
echo "  RUN_TAG=$RUN_TAG"
echo "  ARM=$ARM"
echo "  ITER=$ITER"
echo "  TASK_GROUP=$TASK_GROUP"
echo "  MEGATRON_CKPT_ROOT=$MEGATRON_CKPT_ROOT"
echo "  HF_OUT_DIR=$HF_OUT_DIR"
echo "  EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR"
echo

convert_job="$(sbatch --parsable \
    --export=ALL,RUN_TAG="$RUN_TAG",ARM="$ARM",ITER="$ITER",MEGATRON_CKPT_ROOT="$MEGATRON_CKPT_ROOT",HF_OUT_DIR="$HF_OUT_DIR",OUT_ROOT="$OUT_ROOT",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
    --job-name="tohf_${ARM}_${ITER}" \
    "$SCRIPT_DIR/convert_bakeoff_checkpoint_to_hf.sbatch")"

eval_job="$(sbatch --parsable \
    --dependency="afterok:$convert_job" \
    --export=ALL,MODEL_PATH="$HF_OUT_DIR",OUTPUT_DIR="$EVAL_OUTPUT_DIR",TASK_GROUP="$TASK_GROUP" \
    --job-name="eval_${ARM}_${ITER}_${TASK_GROUP}" \
    "$SCRIPT_DIR/run_eval.sbatch")"

echo "Submitted conversion job: $convert_job"
echo "Submitted lm-eval job:     $eval_job"

if [ "${SUBMIT_INTRINSIC:-0}" = "1" ]; then
    EVAL_JSONL="${EVAL_JSONL:?EVAL_JSONL is required when SUBMIT_INTRINSIC=1}"
    if [ ! -s "$EVAL_JSONL" ]; then
        echo "ERROR: EVAL_JSONL does not exist or is empty: $EVAL_JSONL" >&2
        exit 4
    fi
    METRICS_JSON="${METRICS_JSON:-$ARM_EVAL_ROOT/iter_${ITER_PAD}_tokenizer_fair_metrics.json}"
    DIAG_JSON="${DIAG_JSON:-$ARM_EVAL_ROOT/iter_${ITER_PAD}_new_token_diagnostics.json}"

    metrics_job="$(sbatch --parsable \
        --dependency="afterok:$convert_job" \
        --export=ALL,MODEL_PATH="$HF_OUT_DIR",EVAL_JSONL="$EVAL_JSONL",OUTPUT_JSON="$METRICS_JSON",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
        --job-name="bpc_${ARM}_${ITER}" \
        "$SCRIPT_DIR/run_tokenizer_fair_metrics.sbatch")"

    diag_job="$(sbatch --parsable \
        --dependency="afterok:$convert_job" \
        --export=ALL,MODEL_PATH="$HF_OUT_DIR",EVAL_JSONL="$EVAL_JSONL",OUTPUT_JSON="$DIAG_JSON",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
        --job-name="diag_${ARM}_${ITER}" \
        "$SCRIPT_DIR/run_new_token_diagnostics.sbatch")"

    echo "Submitted intrinsic metrics job: $metrics_job"
    echo "Submitted new-token diagnostics: $diag_job"
fi

echo
echo "Monitor:"
echo "  squeue -j $convert_job,$eval_job -o '%.18i %.32j %.2t %.10M %.10l %D %R'"
