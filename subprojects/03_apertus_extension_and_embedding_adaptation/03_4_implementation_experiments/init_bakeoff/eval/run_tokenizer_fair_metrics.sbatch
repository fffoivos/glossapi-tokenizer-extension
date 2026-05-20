#!/usr/bin/env bash
#SBATCH --job-name=tokenizer_fair_metrics
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err
#
# Compute the v0.7 §5.1 primary intrinsic metrics (BPC, NLL/char, NLL/word,
# tokens/word, chars/token, STRR) for one checkpoint on a held-out Greek
# eval slice. These are the tokenizer-fair metrics for comparing Vanilla
# (vocab 131,072) vs ReTok/Centroid (vocab 148,480).
#
# Required env vars (export via --export=...):
#   MODEL_PATH    — HF-format checkpoint dir (e.g. Apertus base, or an arm's checkpoint)
#   EVAL_JSONL    — held-out Greek JSONL (one doc per line: {text, source, register, doc_id})
#   OUTPUT_JSON   — where to write the metrics report
#
# Optional:
#   MAX_CONTEXT   — context-window cap (default 4096, matches bakeoff seq_length)
#   MAX_DOCS      — cap for debugging
#
# Submit:
#   sbatch --export=ALL,MODEL_PATH=...,EVAL_JSONL=...,OUTPUT_JSON=... run_tokenizer_fair_metrics.sbatch

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
EVAL_JSONL="${EVAL_JSONL:?EVAL_JSONL is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
MAX_DOCS_ARG=""
[ -n "${MAX_DOCS:-}" ] && MAX_DOCS_ARG="--max-docs $MAX_DOCS"

UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.6.0:v1}"

mkdir -p "$(dirname "$OUTPUT_JSON")"

echo "=== run_tokenizer_fair_metrics.sbatch ==="
date -u
echo "host: $(hostname)"
echo "MODEL_PATH:  $MODEL_PATH"
echo "EVAL_JSONL:  $EVAL_JSONL"
echo "OUTPUT_JSON: $OUTPUT_JSON"
echo "MAX_CONTEXT: $MAX_CONTEXT"
echo

SCRIPT_DIR="$(dirname "$0")"

uenv run "$UENV_IMAGE" --view=default -- bash -c "
set -euo pipefail
pip install --quiet transformers torch 2>&1 | tail -3 || true
python3 $SCRIPT_DIR/compute_tokenizer_fair_metrics.py \
    --model-path '$MODEL_PATH' \
    --eval-jsonl '$EVAL_JSONL' \
    --output-json '$OUTPUT_JSON' \
    --max-context $MAX_CONTEXT \
    --device cuda \
    --dtype bfloat16 \
    $MAX_DOCS_ARG
"

echo
echo "=== done ==="
date -u
echo "metrics written to: $OUTPUT_JSON"
