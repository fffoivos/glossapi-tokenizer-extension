#!/usr/bin/env bash
# CPU-only HPLT cleaning review-queue builder.

#SBATCH --job-name=05_hplt_review_queue
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
require_cpu_only_slurm "05 HPLT review queue"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
SOURCE_ACTION_RUN_DIR="${SOURCE_ACTION_RUN_DIR:-$RUN_ROOT/05_hplt_action_manifest_refined_20260605T193125Z}"
ACTION_MANIFEST="${ACTION_MANIFEST:-$SOURCE_ACTION_RUN_DIR/reports/action_manifest/action_manifest_20260605T193127Z.jsonl}"
RUN_TAG="${RUN_TAG:-05_hplt_review_queue_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/review_queue}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"

TOP_PER_ERROR="${TOP_PER_ERROR:-8}"
RANDOM_PER_ERROR="${RANDOM_PER_ERROR:-12}"
TOP_DESTRUCTIVE="${TOP_DESTRUCTIVE:-40}"
RANDOM_DESTRUCTIVE="${RANDOM_DESTRUCTIVE:-40}"
FALSE_NEGATIVE_CONTROLS="${FALSE_NEGATIVE_CONTROLS:-100}"
BORDERLINE_CONTROLS="${BORDERLINE_CONTROLS:-60}"
SEED="${SEED:-20260605}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR"

echo "=== run_review_queue.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "ACTION_MANIFEST: $ACTION_MANIFEST"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "TOP_PER_ERROR: $TOP_PER_ERROR"
echo "RANDOM_PER_ERROR: $RANDOM_PER_ERROR"
echo "TOP_DESTRUCTIVE: $TOP_DESTRUCTIVE"
echo "RANDOM_DESTRUCTIVE: $RANDOM_DESTRUCTIVE"
echo "FALSE_NEGATIVE_CONTROLS: $FALSE_NEGATIVE_CONTROLS"
echo "BORDERLINE_CONTROLS: $BORDERLINE_CONTROLS"
echo "SEED: $SEED"
echo

test -s "$ACTION_MANIFEST"
test -f "$CLEANING_DIR/scripts/build_review_queue.py"

python3 "$CLEANING_DIR/scripts/build_review_queue.py" \
  --action-manifest "$ACTION_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --top-per-error "$TOP_PER_ERROR" \
  --random-per-error "$RANDOM_PER_ERROR" \
  --top-destructive "$TOP_DESTRUCTIVE" \
  --random-destructive "$RANDOM_DESTRUCTIVE" \
  --false-negative-controls "$FALSE_NEGATIVE_CONTROLS" \
  --borderline-controls "$BORDERLINE_CONTROLS" \
  --seed "$SEED"

LATEST_SUMMARY_JSON="$(ls -1t "$OUTPUT_DIR"/review_queue_summary_*.json | head -1)"
LATEST_SUMMARY_MD="$(ls -1t "$OUTPUT_DIR"/review_queue_*.md | head -1)"
LATEST_QUEUE_JSONL="$(ls -1t "$OUTPUT_DIR"/review_queue_*.jsonl | head -1)"
LATEST_QUEUE_CSV="$(ls -1t "$OUTPUT_DIR"/review_queue_*.csv | head -1)"
cp "$LATEST_SUMMARY_JSON" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_QUEUE_CSV" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_review_queue_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
ACTION_MANIFEST="$ACTION_MANIFEST"
OUTPUT_DIR="$OUTPUT_DIR"
REVIEW_QUEUE_JSONL="$LATEST_QUEUE_JSONL"
REVIEW_QUEUE_CSV="$LATEST_QUEUE_CSV"
SUMMARY_JSON="$LATEST_SUMMARY_JSON"
SUMMARY_MD="$LATEST_SUMMARY_MD"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
SEED="$SEED"
PATHS

echo
echo "=== HPLT review queue complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "REVIEW_QUEUE_JSONL=$LATEST_QUEUE_JSONL"
echo "REVIEW_QUEUE_CSV=$LATEST_QUEUE_CSV"
echo "SUMMARY_JSON=$LATEST_SUMMARY_JSON"
echo "SUMMARY_MD=$LATEST_SUMMARY_MD"
