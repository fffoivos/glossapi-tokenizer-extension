#!/usr/bin/env bash
# CPU-only boundary-spec review-pack materializer for unresolved HPLT actions.

#SBATCH --job-name=05_hplt_boundary_pack
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
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
require_cpu_only_slurm "05 HPLT boundary-spec review pack"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
RUN_TAG="${RUN_TAG:-05_hplt_boundary_pack_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/boundary_spec_review_pack}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
PACK_NAME="${PACK_NAME:-boundary_spec_review_pack_$(date -u +%Y%m%dT%H%M%SZ)}"
INPUT_PARQUET="${INPUT_PARQUET:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet}"
TOKENIZER_JSON="${TOKENIZER_JSON:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/tokenizer.json}"
XFER_VENV="${XFER_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/hplt_cleaning_py311_xfer}"
MARKDOWN_CHARS="${MARKDOWN_CHARS:-6000}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_boundary_spec_review_pack.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "INPUT_PARQUET: $INPUT_PARQUET"
echo "TOKENIZER_JSON: $TOKENIZER_JSON"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PACK_NAME: $PACK_NAME"
echo "MARKDOWN_CHARS: $MARKDOWN_CHARS"
echo

test -s "$INPUT_PARQUET"
test -s "$TOKENIZER_JSON"
test -f "$CLEANING_DIR/scripts/build_boundary_spec_review_pack.py"

if [[ ! -x "$XFER_VENV/bin/python" ]]; then
  python3.11 -m venv "$XFER_VENV"
fi

# pyarrow is already present in the xfer venv from prior jobs; tokenizers may
# be missing on a fresh venv.
"$XFER_VENV/bin/python" - <<'PY'
import importlib.util
import subprocess
import sys

missing = [name for name in ("pyarrow", "tokenizers") if importlib.util.find_spec(name) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
PY

PYTHON="$XFER_VENV/bin/python"
"$PYTHON" - <<'PY'
import pyarrow
import tokenizers
print("pyarrow", pyarrow.__version__)
print("tokenizers", tokenizers.__version__)
PY

"$PYTHON" "$CLEANING_DIR/scripts/build_boundary_spec_review_pack.py" \
  --reports-dir "$CLEANING_DIR/reports" \
  --input-parquet "$INPUT_PARQUET" \
  --tokenizer-json "$TOKENIZER_JSON" \
  --output-prefix "$OUTPUT_DIR/$PACK_NAME" \
  --markdown-chars "$MARKDOWN_CHARS"

LATEST_JSONL="$OUTPUT_DIR/$PACK_NAME.jsonl"
LATEST_TEMPLATE="$OUTPUT_DIR/${PACK_NAME}_annotation_template.jsonl"
LATEST_MD="$OUTPUT_DIR/$PACK_NAME.md"
LATEST_SUMMARY="$OUTPUT_DIR/${PACK_NAME}_summary.json"

test -f "$LATEST_JSONL"
test -f "$LATEST_TEMPLATE"
test -s "$LATEST_MD"
test -s "$LATEST_SUMMARY"

cp "$LATEST_JSONL" "$REPORT_COPY_DIR/"
cp "$LATEST_TEMPLATE" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_boundary_spec_review_pack_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
INPUT_PARQUET="$INPUT_PARQUET"
TOKENIZER_JSON="$TOKENIZER_JSON"
OUTPUT_DIR="$OUTPUT_DIR"
BOUNDARY_REVIEW_PACK_JSONL="$LATEST_JSONL"
BOUNDARY_ANNOTATION_TEMPLATE_JSONL="$LATEST_TEMPLATE"
BOUNDARY_REVIEW_PACK_MD="$LATEST_MD"
SUMMARY_JSON="$LATEST_SUMMARY"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
PACK_NAME="$PACK_NAME"
MARKDOWN_CHARS="$MARKDOWN_CHARS"
PATHS

echo
echo "=== HPLT boundary-spec review pack complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "BOUNDARY_REVIEW_PACK_JSONL=$LATEST_JSONL"
echo "BOUNDARY_ANNOTATION_TEMPLATE_JSONL=$LATEST_TEMPLATE"
echo "BOUNDARY_REVIEW_PACK_MD=$LATEST_MD"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
