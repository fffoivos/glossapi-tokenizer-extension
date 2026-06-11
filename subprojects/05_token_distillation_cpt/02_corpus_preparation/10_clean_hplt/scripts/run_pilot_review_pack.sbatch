#!/usr/bin/env bash
# CPU-only pilot review-pack builder for HPLT cleaning.

#SBATCH --job-name=05_hplt_pilot_pack
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
require_cpu_only_slurm "05 HPLT pilot review pack"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
SOURCE_REVIEW_QUEUE_RUN_DIR="${SOURCE_REVIEW_QUEUE_RUN_DIR:-$RUN_ROOT/05_hplt_review_queue_20260605T193634Z}"
REVIEW_QUEUE="${REVIEW_QUEUE:-$SOURCE_REVIEW_QUEUE_RUN_DIR/reports/review_queue/review_queue_20260605T193634Z.jsonl}"
RUN_TAG="${RUN_TAG:-05_hplt_pilot_review_pack_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/pilot_review}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
SEED="${SEED:-20260605}"
TARGET_PROFILE="${TARGET_PROFILE:-legacy}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR"

echo "=== run_pilot_review_pack.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "REVIEW_QUEUE: $REVIEW_QUEUE"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "SEED: $SEED"
echo "TARGET_PROFILE: $TARGET_PROFILE"
echo

test -s "$REVIEW_QUEUE"
test -f "$CLEANING_DIR/scripts/build_pilot_review_pack.py"

python3 "$CLEANING_DIR/scripts/build_pilot_review_pack.py" \
  --review-queue "$REVIEW_QUEUE" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$SEED" \
  --target-profile "$TARGET_PROFILE"

LATEST_SUMMARY_JSON="$(ls -1t "$OUTPUT_DIR"/pilot_review_pack_summary_*.json | head -1)"
LATEST_PACK_MD="$(ls -1t "$OUTPUT_DIR"/pilot_review_pack_*.md | head -1)"
LATEST_SEED_JSONL="$(ls -1t "$OUTPUT_DIR"/pilot_review_seed_*.jsonl | head -1)"
cp "$LATEST_SUMMARY_JSON" "$REPORT_COPY_DIR/"
cp "$LATEST_PACK_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_SEED_JSONL" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_pilot_review_pack_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
REVIEW_QUEUE="$REVIEW_QUEUE"
OUTPUT_DIR="$OUTPUT_DIR"
PILOT_PACK_MD="$LATEST_PACK_MD"
PILOT_SEED_JSONL="$LATEST_SEED_JSONL"
SUMMARY_JSON="$LATEST_SUMMARY_JSON"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
SEED="$SEED"
TARGET_PROFILE="$TARGET_PROFILE"
PATHS

echo
echo "=== HPLT pilot review pack complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "PILOT_PACK_MD=$LATEST_PACK_MD"
echo "PILOT_SEED_JSONL=$LATEST_SEED_JSONL"
echo "SUMMARY_JSON=$LATEST_SUMMARY_JSON"
