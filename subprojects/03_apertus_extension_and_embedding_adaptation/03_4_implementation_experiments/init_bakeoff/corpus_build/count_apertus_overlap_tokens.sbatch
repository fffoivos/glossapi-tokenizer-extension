#!/usr/bin/env bash
#SBATCH --job-name=count_overlap_tokens
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/count_overlap_tokens_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/count_overlap_tokens_%j.err
#
# CPU-only token audit for the Apertus-overlap hard-drop docs.
#
# This is intentionally on xfer. It must not consume a GPU node for a text
# scan/token-count job.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/iopsstor/scratch/cscs/fffoivos/repo}"
INIT_BAKEOFF_DIR="$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff"
CORPUS_BUILD_DIR="$INIT_BAKEOFF_DIR/corpus_build"
STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
NANOCHAT_DIR="${NANOCHAT_DIR:-$STAGE_ROOT/nanochat}"
OVERLAY_DIR="${OVERLAY_DIR:-$STAGE_ROOT/apertus_overlap_overlay}"
DROP_DOCS="${DROP_DOCS:-$OVERLAY_DIR/artifacts/dedup_20260519T010924Z/cpt_final_overlay/apertus_overlap_drop_docs.parquet}"
OUT_DIR="${OUT_DIR:-$STAGE_ROOT/token_count_audit_20260526/apertus_overlap_hard_drop}"
BASE_TOKENIZER="${BASE_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
EXT_TOKENIZER="${EXT_TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/preprocess "$OUT_DIR"

source "$INIT_BAKEOFF_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "count_apertus_overlap_tokens"

if [ ! -d "$NANOCHAT_DIR/data" ] || [ ! -f "$DROP_DOCS" ]; then
    echo "Inputs missing; hydrating nanochat source corpus and Apertus overlay." >&2
    echo "This needs existing Hugging Face auth on Clariden." >&2
    bash "$CORPUS_BUILD_DIR/pull_greek_corpus.sh"
fi

if [ ! -d "$NANOCHAT_DIR/data" ]; then
    echo "ERROR: source parquet directory missing: $NANOCHAT_DIR/data" >&2
    exit 2
fi
if [ ! -f "$DROP_DOCS" ]; then
    echo "ERROR: drop-doc parquet missing: $DROP_DOCS" >&2
    exit 2
fi
if [ ! -d "$BASE_TOKENIZER" ]; then
    echo "ERROR: base tokenizer missing: $BASE_TOKENIZER" >&2
    exit 2
fi
if [ ! -d "$EXT_TOKENIZER" ]; then
    echo "ERROR: extended tokenizer missing: $EXT_TOKENIZER" >&2
    exit 2
fi

uenv start pytorch/v2.9.1:v2 --view=default --ignore-tty <<INNER
set -euo pipefail
export PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval:$REPO_DIR
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

python3 -u "$CORPUS_BUILD_DIR/count_apertus_overlap_tokens.py" \
  --source-glob "$NANOCHAT_DIR/data/*.parquet" \
  --drop-docs "$DROP_DOCS" \
  --tokenizer "base=$BASE_TOKENIZER" \
  --tokenizer "extended=$EXT_TOKENIZER" \
  --output-dir "$OUT_DIR" \
  --append-eod
INNER

echo "=== token audit output ==="
find "$OUT_DIR" -maxdepth 1 -type f -print -exec ls -lh {} \;
