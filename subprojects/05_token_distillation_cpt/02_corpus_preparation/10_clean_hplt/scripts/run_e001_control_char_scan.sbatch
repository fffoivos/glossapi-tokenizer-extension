#!/usr/bin/env bash
# CPU-only E001 replacement/control/private-use character scan.

#SBATCH --job-name=05_hplt_e001_scan
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
CLEANING_DIR="$SUBPROJECT_DIR/02_corpus_preparation/10_clean_hplt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 HPLT E001 control-char scan"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
RUN_TAG="${RUN_TAG:-05_hplt_e001_control_char_scan_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="$RUN_DIR/reports/e001_control_char_scan"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
INPUT_PATH="${INPUT_PATH:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet}"
SOURCE_DATASET="${SOURCE_DATASET:-HPLT/ell_Grek_ge8_no_mt_clean60}"
PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-/iopsstor/scratch/cscs/fffoivos/python_envs/hplt_cleaning_py311_xfer}"
PYTHON_BIN="${PYTHON_BIN:-$PYTHON_ENV_DIR/bin/python}"
SETUP_PYTHON_BIN="${SETUP_PYTHON_BIN:-$(command -v python3.11 || true)}"
if [ -z "$SETUP_PYTHON_BIN" ] && [ -x /usr/bin/python3.11 ]; then
  SETUP_PYTHON_BIN=/usr/bin/python3.11
fi

BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_ROWS="${MAX_ROWS:-0}"
TOP_K="${TOP_K:-80}"
RANDOM_SAMPLE_SIZE="${RANDOM_SAMPLE_SIZE:-80}"
SEED="${SEED:-20260606}"
PROGRESS_EVERY_ROWS="${PROGRESS_EVERY_ROWS:-100000}"
RELEASE_MEMORY_EVERY_BATCHES="${RELEASE_MEMORY_EVERY_BATCHES:-50}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_ROOT"

echo "=== run_e001_control_char_scan.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "RUN_DIR: $RUN_DIR"
echo "INPUT_PATH: $INPUT_PATH"
echo "SOURCE_DATASET: $SOURCE_DATASET"
echo "PYTHON_ENV_DIR: $PYTHON_ENV_DIR"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "SETUP_PYTHON_BIN: ${SETUP_PYTHON_BIN:-none}"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_ROWS: $MAX_ROWS"
echo "TOP_K: $TOP_K"
echo "RANDOM_SAMPLE_SIZE: $RANDOM_SAMPLE_SIZE"
echo "SEED: $SEED"
echo "RELEASE_MEMORY_EVERY_BATCHES: $RELEASE_MEMORY_EVERY_BATCHES"
echo

test -s "$INPUT_PATH"
test -f "$CLEANING_DIR/scripts/scan_e001_control_chars.py"
if [ ! -x "$PYTHON_BIN" ]; then
  if [ -z "$SETUP_PYTHON_BIN" ] || [ ! -x "$SETUP_PYTHON_BIN" ]; then
    echo "ERROR: cannot create xfer venv; SETUP_PYTHON_BIN=${SETUP_PYTHON_BIN:-none}" >&2
    exit 127
  fi
  echo "Creating xfer Python venv: $PYTHON_ENV_DIR"
  "$SETUP_PYTHON_BIN" -m venv "$PYTHON_ENV_DIR"
  "$PYTHON_BIN" -m pip install --upgrade pip
fi

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import pyarrow
PY
then
  echo "Installing pyarrow into $PYTHON_ENV_DIR"
  "$PYTHON_BIN" -m pip install 'pyarrow>=17'
fi

run_python() {
  "$PYTHON_BIN" "$@"
}

run_python - <<'PY'
import sys
import pyarrow
print("python", sys.version.split()[0], "pyarrow", pyarrow.__version__)
PY

run_python "$CLEANING_DIR/scripts/scan_e001_control_chars.py" \
  --input "$INPUT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --source-dataset "$SOURCE_DATASET" \
  --batch-size "$BATCH_SIZE" \
  --max-rows "$MAX_ROWS" \
  --top-k "$TOP_K" \
  --random-sample-size "$RANDOM_SAMPLE_SIZE" \
  --seed "$SEED" \
  --progress-every-rows "$PROGRESS_EVERY_ROWS" \
  --release-memory-every-batches "$RELEASE_MEMORY_EVERY_BATCHES"

LATEST_SUMMARY="$(ls -1t "$OUTPUT_DIR"/e001_control_char_scan_*_summary.json | head -1)"
LATEST_PACK_JSONL="$(ls -1t "$OUTPUT_DIR"/e001_control_char_scan_*_review_pack.jsonl | head -1)"
LATEST_PACK_MD="$(ls -1t "$OUTPUT_DIR"/e001_control_char_scan_*_review_pack.md | head -1)"
cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"
cp "$LATEST_PACK_JSONL" "$REPORT_COPY_DIR/"
cp "$LATEST_PACK_MD" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/e001_control_char_scan_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
INPUT_PATH="$INPUT_PATH"
SOURCE_DATASET="$SOURCE_DATASET"
OUTPUT_DIR="$OUTPUT_DIR"
SUMMARY_JSON="$LATEST_SUMMARY"
REVIEW_PACK_JSONL="$LATEST_PACK_JSONL"
REVIEW_PACK_MD="$LATEST_PACK_MD"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
MAX_ROWS="$MAX_ROWS"
TOP_K="$TOP_K"
RANDOM_SAMPLE_SIZE="$RANDOM_SAMPLE_SIZE"
SEED="$SEED"
RELEASE_MEMORY_EVERY_BATCHES="$RELEASE_MEMORY_EVERY_BATCHES"
PATHS

echo
echo "=== E001 control-char scan complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
echo "REVIEW_PACK_JSONL=$LATEST_PACK_JSONL"
echo "REVIEW_PACK_MD=$LATEST_PACK_MD"
