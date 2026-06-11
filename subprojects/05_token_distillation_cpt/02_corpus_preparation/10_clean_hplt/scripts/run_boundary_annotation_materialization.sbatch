#!/usr/bin/env bash
# CPU-only materializer for reviewed HPLT boundary annotations.

#SBATCH --job-name=05_hplt_boundary_apply
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
CLEANING_DIR="$SUBPROJECT_DIR/02_corpus_preparation/10_clean_hplt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 HPLT boundary annotation materialization"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
RUN_TAG="${RUN_TAG:-05_hplt_boundary_apply_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/boundary_annotation_materialization}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
PACK_NAME="${PACK_NAME:-boundary_annotation_materialization_$(date -u +%Y%m%dT%H%M%SZ)}"
REVIEW_PACK="${REVIEW_PACK:-$CLEANING_DIR/reports/boundary_spec_review_pack_20260606T002713Z.jsonl}"
ANNOTATIONS="${ANNOTATIONS:-$CLEANING_DIR/reports/boundary_spec_review_pack_20260606T002713Z_annotation_template.jsonl}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/tokenizer.json}"
XFER_VENV="${XFER_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/hplt_cleaning_py311_xfer}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_boundary_annotation_materialization.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "REVIEW_PACK: $REVIEW_PACK"
echo "ANNOTATIONS: $ANNOTATIONS"
echo "TOKENIZER_JSON: $TOKENIZER_JSON"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PACK_NAME: $PACK_NAME"
echo

test -s "$REVIEW_PACK"
test -s "$ANNOTATIONS"
test -f "$CLEANING_DIR/scripts/materialize_boundary_annotations.py"

if [[ ! -x "$XFER_VENV/bin/python" ]]; then
  python3.11 -m venv "$XFER_VENV"
fi

"$XFER_VENV/bin/python" - <<'PY'
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("tokenizers") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tokenizers"])
PY

PYTHON="$XFER_VENV/bin/python"
"$PYTHON" - <<'PY'
import tokenizers
print("tokenizers", tokenizers.__version__)
PY

"$PYTHON" "$CLEANING_DIR/scripts/materialize_boundary_annotations.py" \
  --review-pack "$REVIEW_PACK" \
  --annotations "$ANNOTATIONS" \
  --output-prefix "$OUTPUT_DIR/$PACK_NAME" \
  --tokenizer-json "$TOKENIZER_JSON"

LATEST_DECISION="$OUTPUT_DIR/${PACK_NAME}_decision_manifest.jsonl"
LATEST_SHADOW="$OUTPUT_DIR/${PACK_NAME}_shadow_records.jsonl"
LATEST_ISSUES="$OUTPUT_DIR/${PACK_NAME}_issues.jsonl"
LATEST_SUMMARY="$OUTPUT_DIR/${PACK_NAME}_summary.json"
LATEST_MD="$OUTPUT_DIR/$PACK_NAME.md"

test -f "$LATEST_DECISION"
test -f "$LATEST_SHADOW"
test -f "$LATEST_ISSUES"
test -s "$LATEST_SUMMARY"
test -s "$LATEST_MD"

cp "$LATEST_DECISION" "$REPORT_COPY_DIR/"
cp "$LATEST_SHADOW" "$REPORT_COPY_DIR/"
cp "$LATEST_ISSUES" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_boundary_annotation_materialization_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
REVIEW_PACK="$REVIEW_PACK"
ANNOTATIONS="$ANNOTATIONS"
TOKENIZER_JSON="$TOKENIZER_JSON"
OUTPUT_DIR="$OUTPUT_DIR"
DECISION_MANIFEST_JSONL="$LATEST_DECISION"
SHADOW_RECORDS_JSONL="$LATEST_SHADOW"
ISSUES_JSONL="$LATEST_ISSUES"
SUMMARY_JSON="$LATEST_SUMMARY"
MARKDOWN="$LATEST_MD"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
PACK_NAME="$PACK_NAME"
PATHS

echo
echo "=== HPLT boundary annotation materialization complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "DECISION_MANIFEST_JSONL=$LATEST_DECISION"
echo "SHADOW_RECORDS_JSONL=$LATEST_SHADOW"
echo "ISSUES_JSONL=$LATEST_ISSUES"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
echo "MARKDOWN=$LATEST_MD"
