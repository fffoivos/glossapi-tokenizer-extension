#!/usr/bin/env bash
# CPU-only reviewed HPLT shadow-overlay aggregation.

#SBATCH --job-name=05_hplt_reviewed_overlay
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
require_cpu_only_slurm "05 HPLT reviewed shadow overlay"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG:-05_hplt_reviewed_overlay_${SLURM_JOB_ID:-manual}_$TIMESTAMP}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/reviewed_shadow_overlay}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
REPORT_NAME="${REPORT_NAME:-reviewed_shadow_overlay_$TIMESTAMP}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_reviewed_shadow_overlay.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "REPORT_NAME: $REPORT_NAME"
echo

test -f "$CLEANING_DIR/scripts/build_reviewed_shadow_overlay.py"
test -f "$CLEANING_DIR/scripts/validate_shadow_manifest.py"

python3.11 -m py_compile \
  "$CLEANING_DIR/scripts/build_reviewed_shadow_overlay.py" \
  "$CLEANING_DIR/scripts/validate_shadow_manifest.py"

python3.11 "$CLEANING_DIR/scripts/build_reviewed_shadow_overlay.py" \
  --reports-dir "$CLEANING_DIR/reports" \
  --output-prefix "$OUTPUT_DIR/$REPORT_NAME" \
  --timestamp "$TIMESTAMP" \
  --fail-on-issues

LATEST_SUMMARY="$OUTPUT_DIR/${REPORT_NAME}_summary.json"
LATEST_MD="$OUTPUT_DIR/$REPORT_NAME.md"
LATEST_MANIFEST="$OUTPUT_DIR/${REPORT_NAME}_decision_manifest.jsonl"
LATEST_SHADOW="$OUTPUT_DIR/${REPORT_NAME}_shadow_records.jsonl"
LATEST_VALIDATION="$OUTPUT_DIR/${REPORT_NAME}_validation.json"

test -s "$LATEST_SUMMARY"
test -s "$LATEST_MD"
test -s "$LATEST_MANIFEST"
test -s "$LATEST_VALIDATION"

cp "$LATEST_SUMMARY" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_MANIFEST" "$REPORT_COPY_DIR/"
cp "$LATEST_SHADOW" "$REPORT_COPY_DIR/"
cp "$LATEST_VALIDATION" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_reviewed_shadow_overlay_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
OUTPUT_DIR="$OUTPUT_DIR"
REVIEWED_SHADOW_OVERLAY_SUMMARY="$LATEST_SUMMARY"
REVIEWED_SHADOW_OVERLAY_MD="$LATEST_MD"
REVIEWED_SHADOW_OVERLAY_MANIFEST="$LATEST_MANIFEST"
REVIEWED_SHADOW_OVERLAY_SHADOW_RECORDS="$LATEST_SHADOW"
REVIEWED_SHADOW_OVERLAY_VALIDATION="$LATEST_VALIDATION"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
REPORT_NAME="$REPORT_NAME"
TIMESTAMP="$TIMESTAMP"
PATHS

echo
echo "=== HPLT reviewed shadow overlay complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "REVIEWED_SHADOW_OVERLAY_SUMMARY=$LATEST_SUMMARY"
echo "REVIEWED_SHADOW_OVERLAY_MANIFEST=$LATEST_MANIFEST"
echo "REVIEWED_SHADOW_OVERLAY_VALIDATION=$LATEST_VALIDATION"
