#!/usr/bin/env bash
# CPU-only pending-boundary review-pack builder.

#SBATCH --job-name=05_hplt_pending_bounds
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
require_cpu_only_slurm "05 HPLT pending-boundary review pack"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
RUN_TAG="${RUN_TAG:-05_hplt_pending_bounds_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/pending_boundary_review_pack}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
PACK_NAME="${PACK_NAME:-pending_boundary_review_pack_$(date -u +%Y%m%dT%H%M%SZ)}"
REVIEW_PACK="${REVIEW_PACK:-$CLEANING_DIR/reports/boundary_spec_review_pack_20260606T002713Z.jsonl}"
RESOLVED_MANIFEST="${RESOLVED_MANIFEST:-$CLEANING_DIR/reports/boundary_whole_doc_materialization_20260606T011700Z_decision_manifest.jsonl}"
RESOLVED_MANIFESTS="${RESOLVED_MANIFESTS:-$RESOLVED_MANIFEST}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_pending_boundary_review_pack.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "REVIEW_PACK: $REVIEW_PACK"
echo "RESOLVED_MANIFESTS: $RESOLVED_MANIFESTS"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "PACK_NAME: $PACK_NAME"
echo

test -s "$REVIEW_PACK"
for manifest in $RESOLVED_MANIFESTS; do
  test -s "$manifest"
done
test -f "$CLEANING_DIR/scripts/build_pending_boundary_review_pack.py"

RESOLVED_ARGS=()
for manifest in $RESOLVED_MANIFESTS; do
  RESOLVED_ARGS+=(--resolved-manifest "$manifest")
done

python3.11 "$CLEANING_DIR/scripts/build_pending_boundary_review_pack.py" \
  --review-pack "$REVIEW_PACK" \
  "${RESOLVED_ARGS[@]}" \
  --output-prefix "$OUTPUT_DIR/$PACK_NAME"

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

cat > "$RUN_DIR/hplt_pending_boundary_review_pack_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
REVIEW_PACK="$REVIEW_PACK"
RESOLVED_MANIFESTS="$RESOLVED_MANIFESTS"
OUTPUT_DIR="$OUTPUT_DIR"
PENDING_BOUNDARY_PACK_JSONL="$LATEST_JSONL"
PENDING_BOUNDARY_ANNOTATION_TEMPLATE_JSONL="$LATEST_TEMPLATE"
PENDING_BOUNDARY_PACK_MD="$LATEST_MD"
SUMMARY_JSON="$LATEST_SUMMARY"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
PACK_NAME="$PACK_NAME"
PATHS

echo
echo "=== HPLT pending-boundary review pack complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "PENDING_BOUNDARY_PACK_JSONL=$LATEST_JSONL"
echo "PENDING_BOUNDARY_ANNOTATION_TEMPLATE_JSONL=$LATEST_TEMPLATE"
echo "PENDING_BOUNDARY_PACK_MD=$LATEST_MD"
echo "SUMMARY_JSON=$LATEST_SUMMARY"
