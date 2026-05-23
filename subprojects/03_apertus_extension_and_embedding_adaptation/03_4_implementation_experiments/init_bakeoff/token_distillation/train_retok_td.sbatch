#!/usr/bin/env bash
#SBATCH --job-name=retok_td_smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.err

set -euo pipefail

STUDENT_MODEL="${STUDENT_MODEL:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/retok}"
BASE_TOKENIZER="${BASE_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
STUDENT_TOKENIZER="${STUDENT_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480}"
COVERAGE_JSONL="${COVERAGE_JSONL:?COVERAGE_JSONL is required}"
SNIPPETS_JSONL="${SNIPPETS_JSONL:?SNIPPETS_JSONL is required}"
TOKEN_IDS_FILE="${TOKEN_IDS_FILE:?TOKEN_IDS_FILE is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

TARGET_LAYER="${TARGET_LAYER:--1}"
SNIPPETS_PER_TOKEN="${SNIPPETS_PER_TOKEN:-25}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MIN_ACCEPTED_SNIPPETS_PER_TOKEN="${MIN_ACCEPTED_SNIPPETS_PER_TOKEN:-$SNIPPETS_PER_TOKEN}"
MIN_TRAINED_TOKEN_FRACTION="${MIN_TRAINED_TOKEN_FRACTION:-0.90}"
MAX_SELECTED_TOKENS="${MAX_SELECTED_TOKENS:-}"
DRY_RUN="${DRY_RUN:-0}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"

MAX_SELECTED_ARG=""
[ -n "$MAX_SELECTED_TOKENS" ] && MAX_SELECTED_ARG="--max-selected-tokens $MAX_SELECTED_TOKENS"
DRY_RUN_ARG=""
[ "$DRY_RUN" = "1" ] && DRY_RUN_ARG="--dry-run"

mkdir -p "$(dirname "$OUTPUT_DIR")" /capstor/scratch/cscs/fffoivos/runs/token_distillation

if [ -n "${SCRIPT_DIR_OVERRIDE:-}" ]; then
    SCRIPT_DIR="$(cd "$SCRIPT_DIR_OVERRIDE" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/train_retok_td.py" ]; then
    SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

echo "=== train_retok_td.sbatch ==="
date -u
echo "host:              $(hostname)"
echo "STUDENT_MODEL:     $STUDENT_MODEL"
echo "BASE_TOKENIZER:    $BASE_TOKENIZER"
echo "STUDENT_TOKENIZER: $STUDENT_TOKENIZER"
echo "COVERAGE_JSONL:    $COVERAGE_JSONL"
echo "SNIPPETS_JSONL:    $SNIPPETS_JSONL"
echo "TOKEN_IDS_FILE:    $TOKEN_IDS_FILE"
echo "OUTPUT_DIR:        $OUTPUT_DIR"
echo "TARGET_LAYER:      $TARGET_LAYER"
echo "SNIPPETS_PER_TOKEN:$SNIPPETS_PER_TOKEN"
echo "BATCH_SIZE:        $BATCH_SIZE"
echo "MIN_ACCEPTED_SNIPPETS_PER_TOKEN:$MIN_ACCEPTED_SNIPPETS_PER_TOKEN"
echo "MIN_TRAINED_TOKEN_FRACTION:$MIN_TRAINED_TOKEN_FRACTION"
echo "DRY_RUN:           $DRY_RUN"
echo

uenv run "$UENV_IMAGE" --view=default -- bash -c "
set -euo pipefail
export PYTHONPATH='$SCRIPT_DIR/external/token-distillation':\${PYTHONPATH:-}
python3 '$SCRIPT_DIR/train_retok_td.py' \
  --student-model '$STUDENT_MODEL' \
  --base-tokenizer '$BASE_TOKENIZER' \
  --student-tokenizer '$STUDENT_TOKENIZER' \
  --coverage-jsonl '$COVERAGE_JSONL' \
  --snippets-jsonl '$SNIPPETS_JSONL' \
  --token-ids-file '$TOKEN_IDS_FILE' \
  --output-dir '$OUTPUT_DIR' \
  --target-layer '$TARGET_LAYER' \
  --snippets-per-token '$SNIPPETS_PER_TOKEN' \
  --min-accepted-snippets-per-token '$MIN_ACCEPTED_SNIPPETS_PER_TOKEN' \
  --min-trained-token-fraction '$MIN_TRAINED_TOKEN_FRACTION' \
  --epochs '$EPOCHS' \
  --batch-size '$BATCH_SIZE' \
  --learning-rate '$LEARNING_RATE' \
  --dtype bfloat16 \
  --device cuda \
  $MAX_SELECTED_ARG $DRY_RUN_ARG
"

echo
echo "=== done ==="
date -u
echo "output: $OUTPUT_DIR"
