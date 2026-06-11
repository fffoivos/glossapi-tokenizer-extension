#!/usr/bin/env bash
# CPU-only HPLT cleaning inventory and review-bundle sample.

#SBATCH --job-name=05_hplt_clean_inv
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
CLEANING_DIR="$SUBPROJECT_DIR/02_corpus_preparation/10_clean_hplt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 HPLT cleaning inventory"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
RUN_TAG="${RUN_TAG:-05_hplt_cleaning_inventory_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
INPUT_PATH="${INPUT_PATH:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet}"
SOURCE_DATASET="${SOURCE_DATASET:-HPLT/ell_Grek_ge8_no_mt_clean60}"
METADATA_SAMPLE_MANIFEST="${METADATA_SAMPLE_MANIFEST:-}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /users/fffoivos/.local/bin/uv ]; then
  UV_BIN=/users/fffoivos/.local/bin/uv
fi

SCREEN_SAMPLE_SIZE="${SCREEN_SAMPLE_SIZE:-100000}"
GLOBAL_RANDOM_SIZE="${GLOBAL_RANDOM_SIZE:-1067}"
PER_BIN_SIZE="${PER_BIN_SIZE:-200}"
TARGETED_PER_FEATURE="${TARGETED_PER_FEATURE:-100}"
BORDERLINE_SIZE="${BORDERLINE_SIZE:-300}"
MAX_METADATA_ROWS="${MAX_METADATA_ROWS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
TEXT_BATCH_SIZE="${TEXT_BATCH_SIZE:-16}"
FILTERED_FETCH_ID_LIMIT="${FILTERED_FETCH_ID_LIMIT:-5000}"
PROGRESS_EVERY_ROWS="${PROGRESS_EVERY_ROWS:-1000000}"
REVIEW_TEXT_MAX_CHARS="${REVIEW_TEXT_MAX_CHARS:-200000}"
FEATURE_TEXT_MAX_CHARS="${FEATURE_TEXT_MAX_CHARS:-100000}"
RELEASE_MEMORY_EVERY_BATCHES="${RELEASE_MEMORY_EVERY_BATCHES:-200}"

REPORT_COPY_DIR="$CLEANING_DIR/reports"
mkdir -p "$RUN_DIR" "$RUN_ROOT" "$REPORT_COPY_DIR" "$CLEANING_DIR/experiments"

echo "=== run_hplt_cleaning_inventory.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "REPO_ROOT: $REPO_ROOT"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "RUN_DIR: $RUN_DIR"
echo "INPUT_PATH: $INPUT_PATH"
echo "SOURCE_DATASET: $SOURCE_DATASET"
echo "METADATA_SAMPLE_MANIFEST: ${METADATA_SAMPLE_MANIFEST:-none}"
echo "UV_BIN: ${UV_BIN:-none}"
echo "SCREEN_SAMPLE_SIZE: $SCREEN_SAMPLE_SIZE"
echo "GLOBAL_RANDOM_SIZE: $GLOBAL_RANDOM_SIZE"
echo "PER_BIN_SIZE: $PER_BIN_SIZE"
echo "TARGETED_PER_FEATURE: $TARGETED_PER_FEATURE"
echo "BORDERLINE_SIZE: $BORDERLINE_SIZE"
echo "MAX_METADATA_ROWS: $MAX_METADATA_ROWS"
echo "TEXT_BATCH_SIZE: $TEXT_BATCH_SIZE"
echo "FILTERED_FETCH_ID_LIMIT: $FILTERED_FETCH_ID_LIMIT"
echo "FEATURE_TEXT_MAX_CHARS: $FEATURE_TEXT_MAX_CHARS"
echo "RELEASE_MEMORY_EVERY_BATCHES: $RELEASE_MEMORY_EVERY_BATCHES"
echo

test -s "$INPUT_PATH"
test -f "$CLEANING_DIR/scripts/hplt_feature_inventory.py"
if [ -z "$UV_BIN" ] || [ ! -x "$UV_BIN" ]; then
  echo "ERROR: uv is required on xfer for Python 3.12 + pyarrow; UV_BIN=${UV_BIN:-none}" >&2
  exit 127
fi

run_python() {
  "$UV_BIN" run --python 3.12 --with 'pyarrow>=17' -- python "$@"
}

run_python - <<'PY'
import sys
import pyarrow
print("python", sys.version.split()[0], "pyarrow", pyarrow.__version__)
PY

EXTRA_ARGS=()
if [ -n "$METADATA_SAMPLE_MANIFEST" ]; then
  test -s "$METADATA_SAMPLE_MANIFEST"
  EXTRA_ARGS+=(--metadata-sample-manifest "$METADATA_SAMPLE_MANIFEST")
fi

run_python "$CLEANING_DIR/scripts/hplt_feature_inventory.py" \
  --input "$INPUT_PATH" \
  --output-dir "$RUN_DIR" \
  --source-dataset "$SOURCE_DATASET" \
  --screen-sample-size "$SCREEN_SAMPLE_SIZE" \
  --global-random-size "$GLOBAL_RANDOM_SIZE" \
  --per-bin-size "$PER_BIN_SIZE" \
  --targeted-per-feature "$TARGETED_PER_FEATURE" \
  --borderline-size "$BORDERLINE_SIZE" \
  --max-metadata-rows "$MAX_METADATA_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --text-batch-size "$TEXT_BATCH_SIZE" \
  --filtered-fetch-id-limit "$FILTERED_FETCH_ID_LIMIT" \
  --progress-every-rows "$PROGRESS_EVERY_ROWS" \
  --review-text-max-chars "$REVIEW_TEXT_MAX_CHARS" \
  --feature-text-max-chars "$FEATURE_TEXT_MAX_CHARS" \
  --release-memory-every-batches "$RELEASE_MEMORY_EVERY_BATCHES" \
  "${EXTRA_ARGS[@]}"

LATEST_REPORT="$(ls -1t "$RUN_DIR"/reports/hplt_cleaning_inventory_*.json | head -1)"
LATEST_FEATURES="$(ls -1t "$RUN_DIR"/reports/hplt_cleaning_sampled_features_*.parquet | head -1)"
cp "$LATEST_REPORT" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_cleaning_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
INPUT_PATH="$INPUT_PATH"
SOURCE_DATASET="$SOURCE_DATASET"
INVENTORY_JSON="$LATEST_REPORT"
SAMPLED_FEATURES_PARQUET="$LATEST_FEATURES"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
SCREEN_SAMPLE_SIZE="$SCREEN_SAMPLE_SIZE"
GLOBAL_RANDOM_SIZE="$GLOBAL_RANDOM_SIZE"
PER_BIN_SIZE="$PER_BIN_SIZE"
TARGETED_PER_FEATURE="$TARGETED_PER_FEATURE"
BORDERLINE_SIZE="$BORDERLINE_SIZE"
METADATA_SAMPLE_MANIFEST="$METADATA_SAMPLE_MANIFEST"
PATHS

echo
echo "=== HPLT cleaning inventory complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "INVENTORY_JSON=$LATEST_REPORT"
echo "SAMPLED_FEATURES_PARQUET=$LATEST_FEATURES"
echo "REPORT_COPY=$REPORT_COPY_DIR/$(basename "$LATEST_REPORT")"
