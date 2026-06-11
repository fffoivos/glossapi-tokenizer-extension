#!/usr/bin/env bash
# CPU-only global-prevalence annotation summary validator.

#SBATCH --job-name=05_hplt_prev_summary
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
require_cpu_only_slurm "05 HPLT global prevalence annotation summary"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG:-05_hplt_prev_summary_${SLURM_JOB_ID:-manual}_$TIMESTAMP}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/global_prevalence_annotation_summary}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
SUMMARY_NAME="${SUMMARY_NAME:-global_prevalence_annotation_summary_$TIMESTAMP}"
REVIEW_PACK="${REVIEW_PACK:-$CLEANING_DIR/reports/global_prevalence_review_pack_20260606T040025Z.jsonl}"
BASE_SUMMARY="${BASE_SUMMARY:-$CLEANING_DIR/reports/global_prevalence_review_pack_20260606T040025Z_summary.json}"
ANNOTATIONS="${ANNOTATIONS:-$CLEANING_DIR/reports/global_prevalence_review_annotations_20260606T041500Z.jsonl}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_global_prevalence_annotation_summary.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "REVIEW_PACK: $REVIEW_PACK"
echo "BASE_SUMMARY: $BASE_SUMMARY"
echo "ANNOTATIONS: $ANNOTATIONS"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SUMMARY_NAME: $SUMMARY_NAME"
echo

test -s "$REVIEW_PACK"
test -s "$BASE_SUMMARY"
for annotation in $ANNOTATIONS; do
  test -s "$annotation"
done
test -f "$CLEANING_DIR/scripts/summarize_global_prevalence_annotations.py"

python3.11 -m py_compile "$CLEANING_DIR/scripts/summarize_global_prevalence_annotations.py"

ANNOTATION_ARGS=()
for annotation in $ANNOTATIONS; do
  ANNOTATION_ARGS+=("$annotation")
done

python3.11 "$CLEANING_DIR/scripts/summarize_global_prevalence_annotations.py" \
  --review-pack "$REVIEW_PACK" \
  --annotations "${ANNOTATION_ARGS[@]}" \
  --base-summary "$BASE_SUMMARY" \
  --output-prefix "$OUTPUT_DIR/$SUMMARY_NAME" \
  --timestamp "$TIMESTAMP" \
  --fail-on-issues

LATEST_SUMMARY="$OUTPUT_DIR/${SUMMARY_NAME}_summary.json"
LATEST_MD="$OUTPUT_DIR/$SUMMARY_NAME.md"

test -s "$LATEST_SUMMARY"
test -s "$LATEST_MD"

cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_global_prevalence_annotation_summary_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
REVIEW_PACK="$REVIEW_PACK"
BASE_SUMMARY="$BASE_SUMMARY"
ANNOTATIONS="$ANNOTATIONS"
OUTPUT_DIR="$OUTPUT_DIR"
SUMMARY_JSON="$LATEST_SUMMARY"
SUMMARY_MD="$LATEST_MD"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
SUMMARY_NAME="$SUMMARY_NAME"
TIMESTAMP="$TIMESTAMP"
PATHS

echo
echo "=== HPLT global prevalence annotation summary complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
echo "SUMMARY_MD=$LATEST_MD"
