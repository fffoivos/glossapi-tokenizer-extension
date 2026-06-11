#!/usr/bin/env bash
# CPU-only candidate action manifest build from an HPLT review bundle.

#SBATCH --job-name=05_hplt_action_manifest
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/iopsstor/scratch/cscs/fffoivos/repo/glossapi-tokenizer-extension}"
SUBPROJECT_DIR="$REPO_ROOT/subprojects/05_token_distillation_cpt"
CLEANING_DIR="$SUBPROJECT_DIR/02_corpus_preparation/10_clean_hplt"
LEGACY_INIT_DIR="${LEGACY_INIT_DIR:-$REPO_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff}"
source "$LEGACY_INIT_DIR/slurm_cpu_only_guard.sh"
require_cpu_only_slurm "05 HPLT review action manifest"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-$RUN_ROOT/05_hplt_cleaning_inventory_resume_20260605T165145Z}"
INPUT_MANIFEST="${INPUT_MANIFEST:-$SOURCE_RUN_DIR/experiments/review_bundle_20260605T165151Z/manifest.jsonl}"
RUN_TAG="${RUN_TAG:-05_hplt_action_manifest_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/action_manifest}"
THRESHOLD="${THRESHOLD:-0.35}"
TOP_EXAMPLES_PER_TYPE="${TOP_EXAMPLES_PER_TYPE:-20}"
EXCERPT_CHARS="${EXCERPT_CHARS:-1000}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR"

echo "=== run_review_action_manifest.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "INPUT_MANIFEST: $INPUT_MANIFEST"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "THRESHOLD: $THRESHOLD"
echo "TOP_EXAMPLES_PER_TYPE: $TOP_EXAMPLES_PER_TYPE"
echo

test -s "$INPUT_MANIFEST"
test -f "$CLEANING_DIR/scripts/build_review_action_manifest.py"

python3 "$CLEANING_DIR/scripts/build_review_action_manifest.py" \
  --manifest "$INPUT_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --threshold "$THRESHOLD" \
  --top-examples-per-type "$TOP_EXAMPLES_PER_TYPE" \
  --excerpt-chars "$EXCERPT_CHARS"

LATEST_SUMMARY_JSON="$(ls -1t "$OUTPUT_DIR"/candidate_issue_summary_*.json | head -1)"
LATEST_SUMMARY_MD="$(ls -1t "$OUTPUT_DIR"/candidate_issue_summary_*.md | head -1)"
LATEST_ACTION_MANIFEST="$(ls -1t "$OUTPUT_DIR"/action_manifest_*.jsonl | head -1)"
cp "$LATEST_SUMMARY_JSON" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY_MD" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_action_manifest_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
SOURCE_RUN_DIR="$SOURCE_RUN_DIR"
INPUT_MANIFEST="$INPUT_MANIFEST"
OUTPUT_DIR="$OUTPUT_DIR"
ACTION_MANIFEST="$LATEST_ACTION_MANIFEST"
SUMMARY_JSON="$LATEST_SUMMARY_JSON"
SUMMARY_MD="$LATEST_SUMMARY_MD"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
THRESHOLD="$THRESHOLD"
TOP_EXAMPLES_PER_TYPE="$TOP_EXAMPLES_PER_TYPE"
PATHS

echo
echo "=== HPLT review action manifest complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "ACTION_MANIFEST=$LATEST_ACTION_MANIFEST"
echo "SUMMARY_JSON=$LATEST_SUMMARY_JSON"
echo "SUMMARY_MD=$LATEST_SUMMARY_MD"
