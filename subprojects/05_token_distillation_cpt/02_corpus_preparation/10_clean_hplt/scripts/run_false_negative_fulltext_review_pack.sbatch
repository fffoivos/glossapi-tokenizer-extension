#!/usr/bin/env bash
# CPU-only full-text review-pack materializer for remaining HPLT FN controls.

#SBATCH --job-name=05_hplt_fn_fulltext
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
CLEANING_DIR="$SUBPROJECT_DIR/02_corpus_preparation/10_clean_hplt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 HPLT FN full-text review pack"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
RUN_TAG="${RUN_TAG:-05_hplt_fn_fulltext_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/false_negative_fulltext_review_pack}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
INPUT_PACK="${INPUT_PACK:-$CLEANING_DIR/reports/false_negative_global_remaining_review_pack_20260606T014500Z.jsonl}"
PACK_NAME="${PACK_NAME:-false_negative_global_remaining_fulltext_review_pack_20260606T020000Z}"
STATUSES="${STATUSES:-needs_review}"
MAX_DOC_CHARS="${MAX_DOC_CHARS:-0}"
MARKDOWN_CHARS="${MARKDOWN_CHARS:-4000}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR"

echo "=== run_false_negative_fulltext_review_pack.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "INPUT_PACK: $INPUT_PACK"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PACK_NAME: $PACK_NAME"
echo "STATUSES: $STATUSES"
echo "MAX_DOC_CHARS: $MAX_DOC_CHARS"
echo "MARKDOWN_CHARS: $MARKDOWN_CHARS"
echo

test -s "$INPUT_PACK"
test -f "$CLEANING_DIR/scripts/build_false_negative_fulltext_review_pack.py"

python3 "$CLEANING_DIR/scripts/build_false_negative_fulltext_review_pack.py" \
  --input-pack "$INPUT_PACK" \
  --output-prefix "$OUTPUT_DIR/$PACK_NAME" \
  --statuses "$STATUSES" \
  --max-doc-chars "$MAX_DOC_CHARS" \
  --markdown-chars "$MARKDOWN_CHARS"

LATEST_JSONL="$OUTPUT_DIR/$PACK_NAME.jsonl"
LATEST_MD="$OUTPUT_DIR/$PACK_NAME.md"
LATEST_SUMMARY="$OUTPUT_DIR/${PACK_NAME}_summary.json"
cp "$LATEST_JSONL" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_fn_fulltext_review_pack_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
INPUT_PACK="$INPUT_PACK"
OUTPUT_DIR="$OUTPUT_DIR"
FULLTEXT_REVIEW_PACK_JSONL="$LATEST_JSONL"
FULLTEXT_REVIEW_PACK_MD="$LATEST_MD"
SUMMARY_JSON="$LATEST_SUMMARY"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
PACK_NAME="$PACK_NAME"
STATUSES="$STATUSES"
MAX_DOC_CHARS="$MAX_DOC_CHARS"
MARKDOWN_CHARS="$MARKDOWN_CHARS"
PATHS

echo
echo "=== HPLT FN full-text review pack complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "FULLTEXT_REVIEW_PACK_JSONL=$LATEST_JSONL"
echo "FULLTEXT_REVIEW_PACK_MD=$LATEST_MD"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
