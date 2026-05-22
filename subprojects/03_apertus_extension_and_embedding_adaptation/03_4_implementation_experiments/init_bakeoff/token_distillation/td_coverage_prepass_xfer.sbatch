#!/usr/bin/env bash
#SBATCH --job-name=td_coverage_prepass
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/iopsstor/scratch/cscs/fffoivos/repo}"
INIT_BAKEOFF_DIR="$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff"
source "$INIT_BAKEOFF_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "td_coverage_prepass"

INPUT_JSONL="${INPUT_JSONL:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.jsonl}"
BASE_TOKENIZER="${BASE_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
STUDENT_TOKENIZER="${STUDENT_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480}"
OUTPUT_DIR="${OUTPUT_DIR:-/iopsstor/scratch/cscs/fffoivos/token_distillation/coverage_2b_modern_$(date -u +%Y%m%dT%H%M%SZ)}"
TARGET_EXTENDED_TOKENS="${TARGET_EXTENDED_TOKENS:-2000000000}"
NEW_ID_START="${NEW_ID_START:-131072}"
NEW_ID_END="${NEW_ID_END:-148480}"
SNIPPETS_PER_TOKEN="${SNIPPETS_PER_TOKEN:-100}"
SNIPPET_TOKEN_RADIUS="${SNIPPET_TOKEN_RADIUS:-50}"
PROGRESS_TOKEN_INTERVAL="${PROGRESS_TOKEN_INTERVAL:-50000000}"
SEED="${SEED:-20260523}"
PYTHON_BIN="${PYTHON_BIN:-/iopsstor/scratch/cscs/fffoivos/python_envs/td_coverage_py311_xfer/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    echo "Create it once on Clariden xfer with Python 3.11 and:" >&2
    echo "  python3.11 -m venv /iopsstor/scratch/cscs/fffoivos/python_envs/td_coverage_py311_xfer" >&2
    echo "  /iopsstor/scratch/cscs/fffoivos/python_envs/td_coverage_py311_xfer/bin/python -m pip install tokenizers==0.22.1" >&2
    exit 127
fi

mkdir -p "$OUTPUT_DIR" /capstor/scratch/cscs/fffoivos/runs/token_distillation

echo "=== td_coverage_prepass_xfer ==="
date -u
echo "host: $HOSTNAME"
echo "INPUT_JSONL:             $INPUT_JSONL"
echo "BASE_TOKENIZER:          $BASE_TOKENIZER"
echo "STUDENT_TOKENIZER:       $STUDENT_TOKENIZER"
echo "OUTPUT_DIR:              $OUTPUT_DIR"
echo "TARGET_EXTENDED_TOKENS:  $TARGET_EXTENDED_TOKENS"
echo "NEW_ID_RANGE:            [$NEW_ID_START, $NEW_ID_END)"
echo "SNIPPETS_PER_TOKEN:      $SNIPPETS_PER_TOKEN"
echo "SNIPPET_TOKEN_RADIUS:    $SNIPPET_TOKEN_RADIUS"
echo "PROGRESS_TOKEN_INTERVAL: $PROGRESS_TOKEN_INTERVAL"
echo "PYTHON_BIN:              $PYTHON_BIN"
echo

"$PYTHON_BIN" -u "$INIT_BAKEOFF_DIR/token_distillation/td_coverage_prepass.py" \
  --input-jsonl "$INPUT_JSONL" \
  --base-tokenizer "$BASE_TOKENIZER" \
  --student-tokenizer "$STUDENT_TOKENIZER" \
  --output-dir "$OUTPUT_DIR" \
  --target-extended-tokens "$TARGET_EXTENDED_TOKENS" \
  --new-id-start "$NEW_ID_START" \
  --new-id-end "$NEW_ID_END" \
  --snippets-per-token "$SNIPPETS_PER_TOKEN" \
  --snippet-token-radius "$SNIPPET_TOKEN_RADIUS" \
  --progress-token-interval "$PROGRESS_TOKEN_INTERVAL" \
  --seed "$SEED"

echo
echo "=== TD coverage prepass complete ==="
find "$OUTPUT_DIR" -maxdepth 2 -type f -printf "%p %s bytes\n" | sort
