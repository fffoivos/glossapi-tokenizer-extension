#!/usr/bin/env bash
# CPU-only serious-FN URL-rescue annotation summary for HPLT cleaning.

#SBATCH --job-name=05_hplt_url_rescue_sum
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
CLEANING_DIR="$SUBPROJECT_DIR/02_corpus_preparation/10_clean_hplt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 HPLT serious FN URL rescue annotation summary"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG:-05_hplt_url_rescue_sum_${SLURM_JOB_ID:-manual}_$TIMESTAMP}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/url_rescue_annotation_summary}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
REVIEW_PACK="${REVIEW_PACK:-$CLEANING_DIR/reports/serious_fn_url_rescue_pack_20260606T060119Z.jsonl}"
ANNOTATIONS="${ANNOTATIONS:-$CLEANING_DIR/reports/serious_fn_url_rescue_annotations_20260606T064500Z.jsonl}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$OUTPUT_DIR/serious_fn_url_rescue_annotation_summary}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_serious_fn_url_rescue_annotation_summary.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "REVIEW_PACK: $REVIEW_PACK"
echo "ANNOTATIONS: $ANNOTATIONS"
echo "OUTPUT_PREFIX: $OUTPUT_PREFIX"
echo

test -s "$REVIEW_PACK"
test -s "$ANNOTATIONS"
test -f "$CLEANING_DIR/scripts/summarize_serious_fn_url_rescue_annotations.py"

python3.11 -m py_compile "$CLEANING_DIR/scripts/summarize_serious_fn_url_rescue_annotations.py"

python3.11 "$CLEANING_DIR/scripts/summarize_serious_fn_url_rescue_annotations.py" \
  --review-pack "$REVIEW_PACK" \
  --annotations "$ANNOTATIONS" \
  --output-prefix "$OUTPUT_PREFIX" \
  --timestamp "$TIMESTAMP"

LATEST_SUMMARY="${OUTPUT_PREFIX}_${TIMESTAMP}_summary.json"
LATEST_MD="${OUTPUT_PREFIX}_${TIMESTAMP}.md"

test -s "$LATEST_SUMMARY"
test -s "$LATEST_MD"

cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_url_rescue_annotation_summary_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
REVIEW_PACK="$REVIEW_PACK"
ANNOTATIONS="$ANNOTATIONS"
SUMMARY_JSON="$LATEST_SUMMARY"
SUMMARY_MD="$LATEST_MD"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
PATHS

echo
echo "=== HPLT serious FN URL rescue annotation summary complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
echo "SUMMARY_MD=$LATEST_MD"
