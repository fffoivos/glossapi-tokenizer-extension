#!/usr/bin/env bash
# CPU-only global-prevalence coverage and next-review pack builder.

#SBATCH --job-name=05_hplt_global_prev
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
require_cpu_only_slurm "05 HPLT global prevalence review pack"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG:-05_hplt_global_prev_${SLURM_JOB_ID:-manual}_$TIMESTAMP}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/global_prevalence_review_pack}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
PACK_NAME="${PACK_NAME:-global_prevalence_review_pack_$TIMESTAMP}"
POLICY_SAMPLE="${POLICY_SAMPLE:-$CLEANING_DIR/reports/policy_evaluation_sample_20260605T210154Z.jsonl}"
PACK_SIZE="${PACK_SIZE:-200}"
FULL_TEXT_CHARS="${FULL_TEXT_CHARS:-0}"
MARKDOWN_CHARS="${MARKDOWN_CHARS:-3000}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_global_prevalence_review_pack.sbatch ==="
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
echo "PACK_SIZE: $PACK_SIZE"
echo

test -s "$POLICY_SAMPLE"
test -f "$CLEANING_DIR/scripts/build_global_prevalence_review_pack.py"
test -f "$CLEANING_DIR/scripts/build_policy_gate_audit.py"

python3.11 -m py_compile \
  "$CLEANING_DIR/scripts/build_global_prevalence_review_pack.py" \
  "$CLEANING_DIR/scripts/build_policy_gate_audit.py"

python3.11 "$CLEANING_DIR/scripts/build_global_prevalence_review_pack.py" \
  --reports-dir "$CLEANING_DIR/reports" \
  --policy-sample "$POLICY_SAMPLE" \
  --output-prefix "$OUTPUT_DIR/$PACK_NAME" \
  --timestamp "$TIMESTAMP" \
  --pack-size "$PACK_SIZE" \
  --full-text-chars "$FULL_TEXT_CHARS" \
  --markdown-chars "$MARKDOWN_CHARS"

LATEST_JSONL="$OUTPUT_DIR/$PACK_NAME.jsonl"
LATEST_TEMPLATE="$OUTPUT_DIR/${PACK_NAME}_annotation_template.jsonl"
LATEST_MD="$OUTPUT_DIR/$PACK_NAME.md"
LATEST_SUMMARY="$OUTPUT_DIR/${PACK_NAME}_summary.json"

test -s "$LATEST_JSONL"
test -s "$LATEST_TEMPLATE"
test -s "$LATEST_MD"
test -s "$LATEST_SUMMARY"

cp "$LATEST_JSONL" "$REPORT_COPY_DIR/"
cp "$LATEST_TEMPLATE" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_global_prevalence_review_pack_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
POLICY_SAMPLE="$POLICY_SAMPLE"
OUTPUT_DIR="$OUTPUT_DIR"
GLOBAL_PREVALENCE_REVIEW_PACK_JSONL="$LATEST_JSONL"
GLOBAL_PREVALENCE_ANNOTATION_TEMPLATE_JSONL="$LATEST_TEMPLATE"
GLOBAL_PREVALENCE_REVIEW_PACK_MD="$LATEST_MD"
SUMMARY_JSON="$LATEST_SUMMARY"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
PACK_NAME="$PACK_NAME"
PACK_SIZE="$PACK_SIZE"
FULL_TEXT_CHARS="$FULL_TEXT_CHARS"
MARKDOWN_CHARS="$MARKDOWN_CHARS"
PATHS

echo
echo "=== HPLT global prevalence review pack complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "GLOBAL_PREVALENCE_REVIEW_PACK_JSONL=$LATEST_JSONL"
echo "GLOBAL_PREVALENCE_ANNOTATION_TEMPLATE_JSONL=$LATEST_TEMPLATE"
echo "GLOBAL_PREVALENCE_REVIEW_PACK_MD=$LATEST_MD"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
