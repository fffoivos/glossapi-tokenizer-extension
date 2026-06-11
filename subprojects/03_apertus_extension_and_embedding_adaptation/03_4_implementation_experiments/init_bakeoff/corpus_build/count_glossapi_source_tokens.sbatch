#!/usr/bin/env bash
#SBATCH --job-name=count_glossapi_src_tokens
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/preprocess/count_glossapi_source_tokens_%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/preprocess/count_glossapi_source_tokens_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/iopsstor/scratch/cscs/fffoivos/repo}"
INIT_BAKEOFF_DIR="$REPO_DIR/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff"
SCRIPT_DIR="$INIT_BAKEOFF_DIR/corpus_build"
STAGE_ROOT="${STAGE_ROOT:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus}"
SOURCE_GLOB="${SOURCE_GLOB:-$STAGE_ROOT/nanochat/data/*.parquet}"
TOKENIZER="${TOKENIZER:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480}"
OUT_DIR="${OUT_DIR:-$STAGE_ROOT/token_count_audit_20260526/glossapi_sources_modern148k}"
PY_ENV="${PY_ENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/preprocess_xfer_py311}"
WORKERS="${WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

mkdir -p /capstor/scratch/cscs/fffoivos/runs/preprocess "$OUT_DIR"

source "$INIT_BAKEOFF_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "count_glossapi_source_tokens"

if ! compgen -G "$SOURCE_GLOB" >/dev/null; then
    echo "ERROR: no parquet files matched: $SOURCE_GLOB" >&2
    exit 2
fi
if [ ! -d "$TOKENIZER" ]; then
    echo "ERROR: tokenizer dir missing: $TOKENIZER" >&2
    exit 2
fi
if [ ! -x "$PY_ENV/bin/python" ]; then
    echo "ERROR: Python env missing: $PY_ENV" >&2
    exit 2
fi

source "$PY_ENV/bin/activate"
export PYTHONPATH=$REPO_DIR
export TOKENIZERS_PARALLELISM=true
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-64}"
THREADS_PER_WORKER="${THREADS_PER_WORKER:-$(( CPUS_PER_TASK / WORKERS ))}"
if [ "$THREADS_PER_WORKER" -lt 1 ]; then
    THREADS_PER_WORKER=1
fi
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export RAYON_NUM_THREADS="$THREADS_PER_WORKER"

echo "cpu layout: workers=$WORKERS cpus_per_task=$CPUS_PER_TASK threads_per_worker=$THREADS_PER_WORKER"

python3 -u "$SCRIPT_DIR/count_source_tokens.py" \
  --source-glob "$SOURCE_GLOB" \
  --exclude-basename-prefix HPLT__ell_Grek_ge8_no_mt_clean60 \
  --tokenizer "$TOKENIZER" \
  --output-dir "$OUT_DIR" \
  --workers "$WORKERS" \
  --batch-size "$BATCH_SIZE"

echo "=== GlossAPI source token-count output ==="
find "$OUT_DIR" -maxdepth 2 -type f | sort | tail -50
