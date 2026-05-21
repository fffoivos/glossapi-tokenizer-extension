#!/usr/bin/env bash
#SBATCH --job-name=new_token_diagnostics
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/eval/%x-%j.err
#
# Compute the v0.7 §5.3 new-token integration diagnostic suite (7 diagnostics)
# at one checkpoint. Designed to be called at every bakeoff checkpoint where
# `compute_tokenizer_fair_metrics` is also called.
#
# Required env vars:
#   MODEL_PATH    — HF-format checkpoint dir
#   EVAL_JSONL    — held-out Greek JSONL (skipped if EMBEDDING_ONLY=1)
#   OUTPUT_JSON   — where to write the diagnostics report
#
# Optional:
#   NEW_ID_LO / NEW_ID_HI   — defaults 131072 / 148480 (modern-only bakeoff scope)
#   BASE_VOCAB_SIZE         — default 131072
#   MAX_CONTEXT             — default 4096
#   MAX_DOCS                — debug cap
#   EMBEDDING_ONLY=1        — skip D1-D5 (saves ~10 min per checkpoint)
#   SKIP_GREEDY=1           — skip just D5

set -euo pipefail

MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
OUTPUT_JSON="${OUTPUT_JSON:?OUTPUT_JSON is required}"
NEW_ID_LO="${NEW_ID_LO:-131072}"
NEW_ID_HI="${NEW_ID_HI:-148480}"
BASE_VOCAB_SIZE="${BASE_VOCAB_SIZE:-131072}"
MAX_CONTEXT="${MAX_CONTEXT:-4096}"
MAX_DOCS_ARG=""
[ -n "${MAX_DOCS:-}" ] && MAX_DOCS_ARG="--max-docs $MAX_DOCS"

EMBED_ONLY_ARG=""
SKIP_GREEDY_ARG=""
EVAL_ARG=""
if [ "${EMBEDDING_ONLY:-0}" = "1" ]; then
    EMBED_ONLY_ARG="--embedding-only"
else
    EVAL_JSONL="${EVAL_JSONL:?EVAL_JSONL is required (unless EMBEDDING_ONLY=1)}"
    EVAL_ARG="--eval-jsonl $EVAL_JSONL"
    [ "${SKIP_GREEDY:-0}" = "1" ] && SKIP_GREEDY_ARG="--skip-greedy"
fi

UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.6.0:v1}"
mkdir -p "$(dirname "$OUTPUT_JSON")"

echo "=== run_new_token_diagnostics.sbatch ==="
date -u
echo "host: $(hostname)"
echo "MODEL_PATH:   $MODEL_PATH"
echo "OUTPUT_JSON:  $OUTPUT_JSON"
echo "NEW_ID_RANGE: [$NEW_ID_LO, $NEW_ID_HI)"
echo "MAX_CONTEXT:  $MAX_CONTEXT"
[ -n "$EMBED_ONLY_ARG" ] && echo "MODE:         embedding-only (D6+D7)"
[ -n "$SKIP_GREEDY_ARG" ] && echo "MODE:         skipping D5 (greedy-gen)"
echo

SCRIPT_DIR="$(dirname "$0")"

uenv run "$UENV_IMAGE" --view=default -- bash -c "
set -euo pipefail
pip install --quiet transformers torch 2>&1 | tail -3 || true
python3 $SCRIPT_DIR/compute_new_token_diagnostics.py \
    --model-path '$MODEL_PATH' \
    $EVAL_ARG \
    --output-json '$OUTPUT_JSON' \
    --new-id-range $NEW_ID_LO $NEW_ID_HI \
    --base-vocab-size $BASE_VOCAB_SIZE \
    --max-context $MAX_CONTEXT \
    --device cuda \
    --dtype bfloat16 \
    $EMBED_ONLY_ARG $SKIP_GREEDY_ARG $MAX_DOCS_ARG
"

echo
echo "=== done ==="
date -u
echo "diagnostics written to: $OUTPUT_JSON"
