#!/usr/bin/env bash
# CPU-only serious-FN URL rescue review-pack builder for HPLT cleaning.

#SBATCH --job-name=05_hplt_url_rescue
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
CLEANING_DIR="$SUBPROJECT_DIR/02_corpus_preparation/10_clean_hplt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 HPLT serious FN URL rescue review pack"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG:-05_hplt_url_rescue_${SLURM_JOB_ID:-manual}_$TIMESTAMP}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/url_rescue_pack}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
POLICY_SAMPLE="${POLICY_SAMPLE:-$CLEANING_DIR/reports/policy_evaluation_sample_20260605T210154Z.jsonl}"
PACK_NAME="${PACK_NAME:-serious_fn_url_rescue_pack}"
MAX_RECORDS="${MAX_RECORDS:-80}"
FULL_TEXT_CHARS="${FULL_TEXT_CHARS:-0}"
MARKDOWN_CHARS="${MARKDOWN_CHARS:-5000}"
URL_THRESHOLD="${URL_THRESHOLD:-1.0}"
SEED="${SEED:-20260606}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_serious_fn_url_rescue_pack.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "POLICY_SAMPLE: $POLICY_SAMPLE"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PACK_NAME: $PACK_NAME"
echo "MAX_RECORDS: $MAX_RECORDS"
echo "URL_THRESHOLD: $URL_THRESHOLD"
echo

test -s "$POLICY_SAMPLE"
test -f "$CLEANING_DIR/scripts/build_serious_fn_url_rescue_pack.py"

python3.11 -m py_compile "$CLEANING_DIR/scripts/build_serious_fn_url_rescue_pack.py"

python3.11 "$CLEANING_DIR/scripts/build_serious_fn_url_rescue_pack.py" \
  --policy-sample "$POLICY_SAMPLE" \
  --reports-dir "$CLEANING_DIR/reports" \
  --output-prefix "$OUTPUT_DIR/$PACK_NAME" \
  --timestamp "$TIMESTAMP" \
  --max-records "$MAX_RECORDS" \
  --full-text-chars "$FULL_TEXT_CHARS" \
  --markdown-chars "$MARKDOWN_CHARS" \
  --url-threshold "$URL_THRESHOLD" \
  --seed "$SEED"

LATEST_JSONL="$OUTPUT_DIR/${PACK_NAME}_${TIMESTAMP}.jsonl"
LATEST_TEMPLATE="$OUTPUT_DIR/${PACK_NAME}_${TIMESTAMP}_annotation_template.jsonl"
LATEST_MD="$OUTPUT_DIR/${PACK_NAME}_${TIMESTAMP}.md"
LATEST_SUMMARY="$OUTPUT_DIR/${PACK_NAME}_${TIMESTAMP}_summary.json"

test -s "$LATEST_JSONL"
test -s "$LATEST_TEMPLATE"
test -s "$LATEST_MD"
test -s "$LATEST_SUMMARY"

cp "$LATEST_JSONL" "$REPORT_COPY_DIR/"
cp "$LATEST_TEMPLATE" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_url_rescue_review_pack_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
POLICY_SAMPLE="$POLICY_SAMPLE"
OUTPUT_DIR="$OUTPUT_DIR"
URL_RESCUE_REVIEW_PACK_JSONL="$LATEST_JSONL"
URL_RESCUE_ANNOTATION_TEMPLATE_JSONL="$LATEST_TEMPLATE"
URL_RESCUE_REVIEW_PACK_MD="$LATEST_MD"
SUMMARY_JSON="$LATEST_SUMMARY"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
PACK_NAME="$PACK_NAME"
MAX_RECORDS="$MAX_RECORDS"
FULL_TEXT_CHARS="$FULL_TEXT_CHARS"
MARKDOWN_CHARS="$MARKDOWN_CHARS"
URL_THRESHOLD="$URL_THRESHOLD"
SEED="$SEED"
PATHS

echo
echo "=== HPLT serious FN URL rescue review pack complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "URL_RESCUE_REVIEW_PACK_JSONL=$LATEST_JSONL"
echo "URL_RESCUE_ANNOTATION_TEMPLATE_JSONL=$LATEST_TEMPLATE"
echo "URL_RESCUE_REVIEW_PACK_MD=$LATEST_MD"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
