#!/usr/bin/env bash
# CPU-only full HPLT rule count and fresh held-out review pack.

#SBATCH --job-name=05_hplt_full_count
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
require_cpu_only_slurm "05 HPLT full rule count heldout"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
RUN_TAG="${RUN_TAG:-05_hplt_full_rule_count_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/full_rule_count}"
INPUT_PATH="${INPUT_PATH:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet}"
SOURCE_DATASET="${SOURCE_DATASET:-HPLT/ell_Grek_ge8_no_mt_clean60}"
THRESHOLD="${THRESHOLD:-0.35}"
MAX_ROWS="${MAX_ROWS:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
FEATURE_TEXT_MAX_CHARS="${FEATURE_TEXT_MAX_CHARS:-100000}"
REVIEW_TEXT_MAX_CHARS="${REVIEW_TEXT_MAX_CHARS:-120000}"
PROGRESS_EVERY_ROWS="${PROGRESS_EVERY_ROWS:-1000000}"
HELDOUT_GLOBAL_N="${HELDOUT_GLOBAL_N:-400}"
HELDOUT_NOHIT_N="${HELDOUT_NOHIT_N:-400}"
HELDOUT_PER_ACTION="${HELDOUT_PER_ACTION:-100}"
HELDOUT_PER_ERROR="${HELDOUT_PER_ERROR:-75}"
HELDOUT_URL_ROUTER_N="${HELDOUT_URL_ROUTER_N:-250}"
HELDOUT_SOURCE_POLICY_N="${HELDOUT_SOURCE_POLICY_N:-150}"
INCLUDE_JSONL="${INCLUDE_JSONL:-}"
INCLUDE_SAMPLE_SET="${INCLUDE_SAMPLE_SET:-}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [ -z "$UV_BIN" ] && [ -x /users/fffoivos/.local/bin/uv ]; then
  UV_BIN=/users/fffoivos/.local/bin/uv
fi

REPORT_COPY_DIR="$CLEANING_DIR/reports"
mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_full_rule_count_heldout.sbatch ==="
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
echo "MAX_ROWS: $MAX_ROWS"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "FEATURE_TEXT_MAX_CHARS: $FEATURE_TEXT_MAX_CHARS"
echo "REVIEW_TEXT_MAX_CHARS: $REVIEW_TEXT_MAX_CHARS"
echo "INCLUDE_JSONL: ${INCLUDE_JSONL:-none}"
echo "INCLUDE_SAMPLE_SET: ${INCLUDE_SAMPLE_SET:-none}"
echo "UV_BIN: ${UV_BIN:-none}"
echo

test -s "$INPUT_PATH"
test -f "$CLEANING_DIR/scripts/build_full_rule_count_and_heldout_pack.py"
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

cd "$CLEANING_DIR"
INCLUDE_ARGS=()
if [ -n "$INCLUDE_JSONL" ]; then
  test -s "$INCLUDE_JSONL"
  INCLUDE_ARGS+=(--include-jsonl "$INCLUDE_JSONL")
  if [ -n "$INCLUDE_SAMPLE_SET" ]; then
    IFS=',' read -r -a INCLUDE_SAMPLE_SET_ARRAY <<< "$INCLUDE_SAMPLE_SET"
    for sample_set in "${INCLUDE_SAMPLE_SET_ARRAY[@]}"; do
      if [ -n "$sample_set" ]; then
        INCLUDE_ARGS+=(--include-sample-set "$sample_set")
      fi
    done
  fi
fi

run_python "$CLEANING_DIR/scripts/build_full_rule_count_and_heldout_pack.py" \
  --input "$INPUT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --source-dataset "$SOURCE_DATASET" \
  --threshold "$THRESHOLD" \
  --max-rows "$MAX_ROWS" \
  --batch-size "$BATCH_SIZE" \
  --feature-text-max-chars "$FEATURE_TEXT_MAX_CHARS" \
  --review-text-max-chars "$REVIEW_TEXT_MAX_CHARS" \
  --progress-every-rows "$PROGRESS_EVERY_ROWS" \
  --heldout-global-n "$HELDOUT_GLOBAL_N" \
  --heldout-nohit-n "$HELDOUT_NOHIT_N" \
  --heldout-per-action "$HELDOUT_PER_ACTION" \
  --heldout-per-error "$HELDOUT_PER_ERROR" \
  --heldout-url-router-n "$HELDOUT_URL_ROUTER_N" \
  --heldout-source-policy-n "$HELDOUT_SOURCE_POLICY_N" \
  --exclude-jsonl-glob "$CLEANING_DIR/reports/*.jsonl" \
  "${INCLUDE_ARGS[@]}"

LATEST_SUMMARY_JSON="$(ls -1t "$OUTPUT_DIR"/full_rule_count_*_summary.json | head -1)"
LATEST_SUMMARY_MD="$(ls -1t "$OUTPUT_DIR"/full_rule_count_*.md | head -1)"
LATEST_PACK="$(ls -1t "$OUTPUT_DIR"/full_rule_count_*_heldout_review_pack.jsonl | head -1)"
LATEST_TEMPLATE="$(ls -1t "$OUTPUT_DIR"/full_rule_count_*_heldout_annotation_template.jsonl | head -1)"
cp "$LATEST_SUMMARY_JSON" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_PACK" "$REPORT_COPY_DIR/"
cp "$LATEST_TEMPLATE" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/full_rule_count_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
OUTPUT_DIR="$OUTPUT_DIR"
INPUT_PATH="$INPUT_PATH"
SOURCE_DATASET="$SOURCE_DATASET"
SUMMARY_JSON="$LATEST_SUMMARY_JSON"
SUMMARY_MD="$LATEST_SUMMARY_MD"
HELDOUT_REVIEW_PACK="$LATEST_PACK"
HELDOUT_ANNOTATION_TEMPLATE="$LATEST_TEMPLATE"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
MAX_ROWS="$MAX_ROWS"
BATCH_SIZE="$BATCH_SIZE"
FEATURE_TEXT_MAX_CHARS="$FEATURE_TEXT_MAX_CHARS"
REVIEW_TEXT_MAX_CHARS="$REVIEW_TEXT_MAX_CHARS"
INCLUDE_JSONL="$INCLUDE_JSONL"
INCLUDE_SAMPLE_SET="$INCLUDE_SAMPLE_SET"
PATHS

echo
echo "=== HPLT full rule count heldout complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "SUMMARY_JSON=$LATEST_SUMMARY_JSON"
echo "SUMMARY_MD=$LATEST_SUMMARY_MD"
echo "HELDOUT_REVIEW_PACK=$LATEST_PACK"
