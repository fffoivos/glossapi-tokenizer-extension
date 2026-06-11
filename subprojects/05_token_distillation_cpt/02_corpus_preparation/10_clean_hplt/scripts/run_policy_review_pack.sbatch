#!/usr/bin/env bash
# CPU-only filtered policy review-pack builder for HPLT cleaning.

#SBATCH --job-name=05_hplt_policy_review_pack
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
require_cpu_only_slurm "05 HPLT policy review pack"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
POLICY_EVAL_RUN_DIR="${POLICY_EVAL_RUN_DIR:-$RUN_ROOT/05_hplt_policy_evaluation_sample_20260605T210154Z}"
POLICY_EVALUATION_SAMPLE="${POLICY_EVALUATION_SAMPLE:-$POLICY_EVAL_RUN_DIR/reports/policy_evaluation_sample/policy_evaluation_sample_20260605T210154Z.jsonl}"
RUN_TAG="${RUN_TAG:-05_hplt_policy_review_pack_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/policy_review_pack}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"

FILTER_ACTIONS="${FILTER_ACTIONS:-drop_doc,normalize_or_trim_span,trim_span}"
FILTER_TASKS="${FILTER_TASKS:-}"
FILTER_STRATA="${FILTER_STRATA:-}"
FILTER_IDS="${FILTER_IDS:-}"
MAX_RECORDS="${MAX_RECORDS:-0}"
MAX_DOC_CHARS="${MAX_DOC_CHARS:-120000}"
PACK_NAME="${PACK_NAME:-destructive_span_policy_review_pack}"
SEED="${SEED:-20260605}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR"

echo "=== run_policy_review_pack.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "POLICY_EVALUATION_SAMPLE: $POLICY_EVALUATION_SAMPLE"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "FILTER_ACTIONS: $FILTER_ACTIONS"
echo "FILTER_TASKS: $FILTER_TASKS"
echo "FILTER_STRATA: $FILTER_STRATA"
echo "FILTER_IDS: $FILTER_IDS"
echo "MAX_RECORDS: $MAX_RECORDS"
echo "MAX_DOC_CHARS: $MAX_DOC_CHARS"
echo "PACK_NAME: $PACK_NAME"
echo "SEED: $SEED"
echo

test -s "$POLICY_EVALUATION_SAMPLE"
test -f "$CLEANING_DIR/scripts/build_policy_review_pack.py"

python3 "$CLEANING_DIR/scripts/build_policy_review_pack.py" \
  --policy-evaluation-sample "$POLICY_EVALUATION_SAMPLE" \
  --output-dir "$OUTPUT_DIR" \
  --actions "$FILTER_ACTIONS" \
  --tasks "$FILTER_TASKS" \
  --strata "$FILTER_STRATA" \
  --ids "$FILTER_IDS" \
  --max-records "$MAX_RECORDS" \
  --max-doc-chars "$MAX_DOC_CHARS" \
  --pack-name "$PACK_NAME" \
  --seed "$SEED"

LATEST_SUMMARY_JSON="$(ls -1t "$OUTPUT_DIR"/*_summary.json | head -1)"
LATEST_PACK_MD="$(ls -1t "$OUTPUT_DIR"/*.md | head -1)"
LATEST_PACK_JSONL="$(ls -1t "$OUTPUT_DIR"/*.jsonl | head -1)"
cp "$LATEST_SUMMARY_JSON" "$REPORT_COPY_DIR/"
cp "$LATEST_PACK_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_PACK_JSONL" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_policy_review_pack_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
POLICY_EVALUATION_SAMPLE="$POLICY_EVALUATION_SAMPLE"
OUTPUT_DIR="$OUTPUT_DIR"
POLICY_REVIEW_PACK_JSONL="$LATEST_PACK_JSONL"
POLICY_REVIEW_PACK_MD="$LATEST_PACK_MD"
SUMMARY_JSON="$LATEST_SUMMARY_JSON"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
FILTER_ACTIONS="$FILTER_ACTIONS"
FILTER_TASKS="$FILTER_TASKS"
FILTER_STRATA="$FILTER_STRATA"
FILTER_IDS="$FILTER_IDS"
MAX_RECORDS="$MAX_RECORDS"
MAX_DOC_CHARS="$MAX_DOC_CHARS"
PACK_NAME="$PACK_NAME"
SEED="$SEED"
PATHS

echo
echo "=== HPLT policy review pack complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "POLICY_REVIEW_PACK_JSONL=$LATEST_PACK_JSONL"
echo "POLICY_REVIEW_PACK_MD=$LATEST_PACK_MD"
echo "SUMMARY_JSON=$LATEST_SUMMARY_JSON"
