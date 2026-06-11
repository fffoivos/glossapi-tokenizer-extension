#!/bin/bash
# CPU-only E001 duplicate/shadow overlay pilot.
#SBATCH --job-name=05_hplt_e001_shadow
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/05_hplt_e001_shadow-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/05_hplt_e001_shadow-%j.err

set -euo pipefail

require_cpu_only_slurm() {
  local label="$1"
  if [ "${SLURM_JOB_PARTITION:-}" != "xfer" ]; then
    echo "ERROR: $label must run on CPU-only xfer, got SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}" >&2
    exit 2
  fi
  if [ -n "${SLURM_JOB_GRES:-}" ] && [ "${SLURM_JOB_GRES:-}" != "none" ]; then
    echo "ERROR: $label requested GRES: ${SLURM_JOB_GRES}" >&2
    exit 2
  fi
  if [ -n "${SLURM_GPUS:-}" ] && [ "${SLURM_GPUS:-}" != "0" ]; then
    echo "ERROR: $label sees SLURM_GPUS=${SLURM_GPUS}" >&2
    exit 2
  fi
}

require_cpu_only_slurm "05 HPLT E001 shadow overlay"

CLEANING_DIR="${CLEANING_DIR:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/02_corpus_preparation/10_clean_hplt}"
RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
RUN_TAG="${RUN_TAG:-05_hplt_e001_shadow_overlay_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="$RUN_DIR/reports/e001_shadow_overlay"
INPUT_PATH="${INPUT_PATH:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet}"
SOURCE_DATASET="${SOURCE_DATASET:-HPLT/ell_Grek_ge8_no_mt_clean60}"
ANNOTATIONS="${ANNOTATIONS:-$CLEANING_DIR/reports/e001_control_char_review_annotations_20260605T232858Z.jsonl}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/tokenizer.json}"
PYTHON_ENV_DIR="${PYTHON_ENV_DIR:-/iopsstor/scratch/cscs/fffoivos/python_envs/hplt_cleaning_py311_xfer}"
SETUP_PYTHON_BIN="${SETUP_PYTHON_BIN:-/usr/bin/python3.11}"
PYTHON_BIN="$PYTHON_ENV_DIR/bin/python"
BATCH_SIZE="${BATCH_SIZE:-256}"

mkdir -p "$RUN_ROOT" "$RUN_DIR" "$OUTPUT_DIR" "$CLEANING_DIR/reports"

echo "=== run_e001_shadow_overlay.sbatch ==="
date -u
echo "host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "RUN_DIR: $RUN_DIR"
echo "INPUT_PATH: $INPUT_PATH"
echo "SOURCE_DATASET: $SOURCE_DATASET"
echo "ANNOTATIONS: $ANNOTATIONS"
echo "TOKENIZER_JSON: $TOKENIZER_JSON"
echo "PYTHON_ENV_DIR: $PYTHON_ENV_DIR"
echo "PYTHON_BIN: $PYTHON_BIN"
echo "SETUP_PYTHON_BIN: $SETUP_PYTHON_BIN"
echo "BATCH_SIZE: $BATCH_SIZE"
echo

if [ ! -x "$PYTHON_BIN" ]; then
  "$SETUP_PYTHON_BIN" -m venv "$PYTHON_ENV_DIR"
fi

"$PYTHON_BIN" - <<'PY'
import importlib.util
import subprocess
import sys

required = ["pyarrow", "tokenizers"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "tokenizers"])
PY

"$PYTHON_BIN" - <<'PY'
import pyarrow
import tokenizers
print("python", __import__("sys").version.split()[0], "pyarrow", pyarrow.__version__, "tokenizers", tokenizers.__version__)
PY

test -f "$CLEANING_DIR/scripts/materialize_e001_shadow_overlay.py"
test -f "$CLEANING_DIR/scripts/validate_shadow_manifest.py"
test -f "$ANNOTATIONS"
test -f "$TOKENIZER_JSON"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
"$PYTHON_BIN" "$CLEANING_DIR/scripts/materialize_e001_shadow_overlay.py" \
  --input "$INPUT_PATH" \
  --annotations "$ANNOTATIONS" \
  --output-dir "$OUTPUT_DIR" \
  --source-dataset "$SOURCE_DATASET" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --timestamp "$TIMESTAMP" \
  --batch-size "$BATCH_SIZE"

LATEST_MANIFEST="$(ls -1t "$OUTPUT_DIR"/e001_shadow_overlay_*_manifest.jsonl | head -1)"
LATEST_SHADOW="$(ls -1t "$OUTPUT_DIR"/e001_shadow_overlay_*_shadow_records.jsonl | head -1)"
LATEST_SUMMARY="$(ls -1t "$OUTPUT_DIR"/e001_shadow_overlay_*_summary.json | head -1)"
LATEST_MD="$(ls -1t "$OUTPUT_DIR"/e001_shadow_overlay_*.md | head -1)"

"$PYTHON_BIN" "$CLEANING_DIR/scripts/validate_shadow_manifest.py" "$LATEST_MANIFEST" --json > "$OUTPUT_DIR/$(basename "${LATEST_MANIFEST%.jsonl}")_validation.json"

cp "$LATEST_MANIFEST" "$CLEANING_DIR/reports/"
cp "$LATEST_SHADOW" "$CLEANING_DIR/reports/"
cp "$LATEST_SUMMARY" "$CLEANING_DIR/reports/"
cp "$LATEST_MD" "$CLEANING_DIR/reports/"
cp "$OUTPUT_DIR/$(basename "${LATEST_MANIFEST%.jsonl}")_validation.json" "$CLEANING_DIR/reports/"

cat > "$RUN_DIR/e001_shadow_overlay_paths.env" <<PATHS
RUN_DIR=$RUN_DIR
OUTPUT_DIR=$OUTPUT_DIR
MANIFEST_JSONL=$LATEST_MANIFEST
SHADOW_RECORDS_JSONL=$LATEST_SHADOW
SUMMARY_JSON=$LATEST_SUMMARY
REPORT_MD=$LATEST_MD
VALIDATION_JSON=$OUTPUT_DIR/$(basename "${LATEST_MANIFEST%.jsonl}")_validation.json
PATHS

echo
echo "=== E001 shadow overlay complete ==="
date -u
cat "$RUN_DIR/e001_shadow_overlay_paths.env"
