#!/usr/bin/env bash
# CPU-only policy-evaluation sample builder for HPLT cleaning.

#SBATCH --job-name=05_hplt_policy_eval_sample
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
require_cpu_only_slurm "05 HPLT policy evaluation sample"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
SOURCE_ACTION_RUN_DIR="${SOURCE_ACTION_RUN_DIR:-$RUN_ROOT/05_hplt_action_manifest_v7_policy_20260605T204452Z}"
ACTION_MANIFEST="${ACTION_MANIFEST:-$SOURCE_ACTION_RUN_DIR/reports/action_manifest/action_manifest_20260605T204455Z.jsonl}"
RUN_TAG="${RUN_TAG:-05_hplt_policy_evaluation_sample_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/policy_evaluation_sample}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"

GLOBAL_RANDOM_N="${GLOBAL_RANDOM_N:-1067}"
WDS_BIN_N="${WDS_BIN_N:-200}"
ACTION_PRECISION_N="${ACTION_PRECISION_N:-60}"
ERROR_PRECISION_N="${ERROR_PRECISION_N:-60}"
FALSE_NEGATIVE_GLOBAL_N="${FALSE_NEGATIVE_GLOBAL_N:-200}"
FALSE_NEGATIVE_BORDERLINE_N="${FALSE_NEGATIVE_BORDERLINE_N:-100}"
HOST_HOTSPOT_COUNT="${HOST_HOTSPOT_COUNT:-20}"
HOST_HOTSPOT_N="${HOST_HOTSPOT_N:-10}"
SEED="${SEED:-20260605}"
PRIOR_ANNOTATIONS="${PRIOR_ANNOTATIONS:-$CLEANING_DIR/reports/pilot_review_annotations_20260605T204701Z.jsonl $CLEANING_DIR/reports/pilot_review_annotations_20260605T202344Z.jsonl $CLEANING_DIR/reports/pilot_review_annotations_20260605T200254Z.jsonl $CLEANING_DIR/reports/pilot_review_annotations_20260605T194002Z.jsonl}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR"

echo "=== run_policy_evaluation_sample.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
echo "SLURM_JOB_PARTITION: ${SLURM_JOB_PARTITION:-none}"
echo "SLURM_JOB_GRES: ${SLURM_JOB_GRES:-none}"
echo "SLURM_GPUS: ${SLURM_GPUS:-none}"
echo "CLEANING_DIR: $CLEANING_DIR"
echo "ACTION_MANIFEST: $ACTION_MANIFEST"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "GLOBAL_RANDOM_N: $GLOBAL_RANDOM_N"
echo "WDS_BIN_N: $WDS_BIN_N"
echo "ACTION_PRECISION_N: $ACTION_PRECISION_N"
echo "ERROR_PRECISION_N: $ERROR_PRECISION_N"
echo "FALSE_NEGATIVE_GLOBAL_N: $FALSE_NEGATIVE_GLOBAL_N"
echo "FALSE_NEGATIVE_BORDERLINE_N: $FALSE_NEGATIVE_BORDERLINE_N"
echo "HOST_HOTSPOT_COUNT: $HOST_HOTSPOT_COUNT"
echo "HOST_HOTSPOT_N: $HOST_HOTSPOT_N"
echo "SEED: $SEED"
echo "PRIOR_ANNOTATIONS: $PRIOR_ANNOTATIONS"
echo

test -s "$ACTION_MANIFEST"
test -f "$CLEANING_DIR/scripts/build_policy_evaluation_sample.py"

PRIOR_ARGS=()
for path in $PRIOR_ANNOTATIONS; do
  if [[ -s "$path" ]]; then
    PRIOR_ARGS+=("$path")
  else
    echo "warning: prior annotation not found or empty: $path"
  fi
done

python3 "$CLEANING_DIR/scripts/build_policy_evaluation_sample.py" \
  --action-manifest "$ACTION_MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --global-random-n "$GLOBAL_RANDOM_N" \
  --wds-bin-n "$WDS_BIN_N" \
  --action-precision-n "$ACTION_PRECISION_N" \
  --error-precision-n "$ERROR_PRECISION_N" \
  --false-negative-global-n "$FALSE_NEGATIVE_GLOBAL_N" \
  --false-negative-borderline-n "$FALSE_NEGATIVE_BORDERLINE_N" \
  --host-hotspot-count "$HOST_HOTSPOT_COUNT" \
  --host-hotspot-n "$HOST_HOTSPOT_N" \
  --seed "$SEED" \
  --prior-annotations "${PRIOR_ARGS[@]}"

LATEST_SUMMARY_JSON="$(ls -1t "$OUTPUT_DIR"/policy_evaluation_sample_summary_*.json | head -1)"
LATEST_SUMMARY_MD="$(ls -1t "$OUTPUT_DIR"/policy_evaluation_sample_*.md | head -1)"
LATEST_SAMPLE_JSONL="$(ls -1t "$OUTPUT_DIR"/policy_evaluation_sample_*.jsonl | head -1)"
LATEST_SAMPLE_CSV="$(ls -1t "$OUTPUT_DIR"/policy_evaluation_sample_*.csv | head -1)"
cp "$LATEST_SUMMARY_JSON" "$REPORT_COPY_DIR/"
cp "$LATEST_SUMMARY_MD" "$REPORT_COPY_DIR/"
cp "$LATEST_SAMPLE_JSONL" "$REPORT_COPY_DIR/"
cp "$LATEST_SAMPLE_CSV" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_policy_evaluation_sample_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
ACTION_MANIFEST="$ACTION_MANIFEST"
OUTPUT_DIR="$OUTPUT_DIR"
POLICY_EVALUATION_SAMPLE_JSONL="$LATEST_SAMPLE_JSONL"
POLICY_EVALUATION_SAMPLE_CSV="$LATEST_SAMPLE_CSV"
SUMMARY_JSON="$LATEST_SUMMARY_JSON"
SUMMARY_MD="$LATEST_SUMMARY_MD"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
SEED="$SEED"
GLOBAL_RANDOM_N="$GLOBAL_RANDOM_N"
WDS_BIN_N="$WDS_BIN_N"
ACTION_PRECISION_N="$ACTION_PRECISION_N"
ERROR_PRECISION_N="$ERROR_PRECISION_N"
FALSE_NEGATIVE_GLOBAL_N="$FALSE_NEGATIVE_GLOBAL_N"
FALSE_NEGATIVE_BORDERLINE_N="$FALSE_NEGATIVE_BORDERLINE_N"
HOST_HOTSPOT_COUNT="$HOST_HOTSPOT_COUNT"
HOST_HOTSPOT_N="$HOST_HOTSPOT_N"
PATHS

echo
echo "=== HPLT policy evaluation sample complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "POLICY_EVALUATION_SAMPLE_JSONL=$LATEST_SAMPLE_JSONL"
echo "POLICY_EVALUATION_SAMPLE_CSV=$LATEST_SAMPLE_CSV"
echo "SUMMARY_JSON=$LATEST_SUMMARY_JSON"
echo "SUMMARY_MD=$LATEST_SUMMARY_MD"
