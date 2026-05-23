#!/usr/bin/env bash
# Submit conversion + eval for one TD layer11 Megatron checkpoint using direct
# checkpoint paths rather than the original bakeoff ${RUN_TAG}_${arm} layout.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <iter> [task-group=full]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ITER="$1"
TASK_GROUP="${2:-full}"

case "$TASK_GROUP" in
    full|retention_only|greek_only|safety_only) ;;
    *) echo "ERROR: task-group must be full|retention_only|greek_only|safety_only, got: $TASK_GROUP" >&2; exit 2 ;;
esac
if ! [[ "$ITER" =~ ^[0-9]+$ ]]; then
    echo "ERROR: iter must be an integer, got: $ITER" >&2
    exit 2
fi

ITER_PAD="$(printf "%07d" "$ITER")"
RUN_TAG="${RUN_TAG:-td_full25_layer11_2b_20260523T165038Z}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/bakeoff}"
MEGATRON_CKPT_ROOT="${MEGATRON_CKPT_ROOT:-$RUN_ROOT/$RUN_TAG/checkpoints}"
HF_TOKENIZER_DIR="${HF_TOKENIZER_DIR:-/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
EVAL_TAG="${EVAL_TAG:-$RUN_TAG}"
EVAL_ROOT="$OUT_ROOT/$EVAL_TAG"
HF_OUT_DIR="${HF_OUT_DIR:-$EVAL_ROOT/iter_${ITER_PAD}_hf}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$EVAL_ROOT/iter_${ITER_PAD}_${TASK_GROUP}}"

if [ ! -d "$MEGATRON_CKPT_ROOT/iter_$ITER_PAD" ]; then
    echo "ERROR: missing checkpoint: $MEGATRON_CKPT_ROOT/iter_$ITER_PAD" >&2
    exit 3
fi
if [ ! -f "$HF_TOKENIZER_DIR/config.json" ] || [ ! -f "$HF_TOKENIZER_DIR/tokenizer.json" ]; then
    echo "ERROR: HF_TOKENIZER_DIR must contain config.json and tokenizer.json: $HF_TOKENIZER_DIR" >&2
    exit 4
fi
if [ "${SUBMIT_INTRINSIC:-0}" = "1" ]; then
    EVAL_JSONL="${EVAL_JSONL:?EVAL_JSONL is required when SUBMIT_INTRINSIC=1}"
    if [ ! -s "$EVAL_JSONL" ]; then
        echo "ERROR: EVAL_JSONL does not exist or is empty: $EVAL_JSONL" >&2
        exit 5
    fi
    METRICS_JSON="${METRICS_JSON:-$EVAL_ROOT/iter_${ITER_PAD}_tokenizer_fair_metrics.json}"
    DIAG_JSON="${DIAG_JSON:-$EVAL_ROOT/iter_${ITER_PAD}_new_token_diagnostics.json}"
fi

mkdir -p "$EVAL_ROOT"

echo "Submitting TD checkpoint eval"
echo "  RUN_TAG=$RUN_TAG"
echo "  ITER=$ITER"
echo "  TASK_GROUP=$TASK_GROUP"
echo "  MEGATRON_CKPT_ROOT=$MEGATRON_CKPT_ROOT"
echo "  HF_TOKENIZER_DIR=$HF_TOKENIZER_DIR"
echo "  HF_OUT_DIR=$HF_OUT_DIR"
echo "  EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR"
echo

convert_job="$(sbatch --parsable \
    --export=ALL,RUN_TAG="$RUN_TAG",ARM=td_layer11,ITER="$ITER",MEGATRON_CKPT_ROOT="$MEGATRON_CKPT_ROOT",HF_TOKENIZER_DIR="$HF_TOKENIZER_DIR",HF_OUT_DIR="$HF_OUT_DIR",OUT_ROOT="$OUT_ROOT",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
    --job-name="tohf_td_l11_${ITER}" \
    "$SCRIPT_DIR/convert_bakeoff_checkpoint_to_hf.sbatch")"

eval_job="$(sbatch --parsable \
    --dependency="afterok:$convert_job" \
    --export=ALL,MODEL_PATH="$HF_OUT_DIR",OUTPUT_DIR="$EVAL_OUTPUT_DIR",TASK_GROUP="$TASK_GROUP" \
    --job-name="eval_td_l11_${ITER}_${TASK_GROUP}" \
    "$SCRIPT_DIR/run_eval.sbatch")"

echo "Submitted conversion job: $convert_job"
echo "Submitted lm-eval job:     $eval_job"

if [ "${SUBMIT_INTRINSIC:-0}" = "1" ]; then
    metrics_job="$(sbatch --parsable \
        --dependency="afterok:$convert_job" \
        --export=ALL,MODEL_PATH="$HF_OUT_DIR",EVAL_JSONL="$EVAL_JSONL",OUTPUT_JSON="$METRICS_JSON",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
        --job-name="bpc_td_l11_${ITER}" \
        "$SCRIPT_DIR/run_tokenizer_fair_metrics.sbatch")"

    diag_job="$(sbatch --parsable \
        --dependency="afterok:$convert_job" \
        --export=ALL,MODEL_PATH="$HF_OUT_DIR",EVAL_JSONL="$EVAL_JSONL",OUTPUT_JSON="$DIAG_JSON",SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR" \
        --job-name="diag_td_l11_${ITER}" \
        "$SCRIPT_DIR/run_new_token_diagnostics.sbatch")"

    echo "Submitted intrinsic metrics job: $metrics_job"
    echo "Submitted new-token diagnostics: $diag_job"
fi
