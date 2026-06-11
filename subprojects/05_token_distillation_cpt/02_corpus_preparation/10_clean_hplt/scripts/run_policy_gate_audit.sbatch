#!/usr/bin/env bash
# CPU-only HPLT cleaning policy-gate audit.

#SBATCH --job-name=05_hplt_policy_gate
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
require_cpu_only_slurm "05 HPLT policy gate audit"

RUN_ROOT="${RUN_ROOT:-/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/hplt_cleaning}"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_TAG="${RUN_TAG:-05_hplt_policy_gate_${SLURM_JOB_ID:-manual}_$TIMESTAMP}"
RUN_DIR="$RUN_ROOT/$RUN_TAG"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_DIR/reports/policy_gate_audit}"
REPORT_COPY_DIR="$CLEANING_DIR/reports"
REPORT_NAME="${REPORT_NAME:-policy_gate_audit_$TIMESTAMP}"

mkdir -p "$OUTPUT_DIR" "$REPORT_COPY_DIR" "$RUN_DIR"

echo "=== run_policy_gate_audit.sbatch ==="
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

test -f "$CLEANING_DIR/scripts/build_policy_gate_audit.py"
python3.11 -m py_compile "$CLEANING_DIR/scripts/build_policy_gate_audit.py"

python3.11 "$CLEANING_DIR/scripts/build_policy_gate_audit.py" \
  --reports-dir "$CLEANING_DIR/reports" \
  --error-catalog "$CLEANING_DIR/ERROR_CATALOG.md" \
  --output-prefix "$OUTPUT_DIR/$REPORT_NAME" \
  --timestamp "$TIMESTAMP"

LATEST_JSON="$OUTPUT_DIR/$REPORT_NAME.json"
LATEST_MD="$OUTPUT_DIR/$REPORT_NAME.md"

test -s "$LATEST_JSON"
test -s "$LATEST_MD"

cp "$LATEST_JSON" "$REPORT_COPY_DIR/"
cp "$LATEST_MD" "$REPORT_COPY_DIR/"

cat > "$RUN_DIR/hplt_policy_gate_audit_paths.env" <<PATHS
RUN_DIR="$RUN_DIR"
OUTPUT_DIR="$OUTPUT_DIR"
POLICY_GATE_AUDIT_JSON="$LATEST_JSON"
POLICY_GATE_AUDIT_MD="$LATEST_MD"
REPORT_COPY_DIR="$REPORT_COPY_DIR"
REPORT_NAME="$REPORT_NAME"
TIMESTAMP="$TIMESTAMP"
PATHS

echo
echo "=== HPLT policy gate audit complete ==="
date -u
echo "RUN_DIR=$RUN_DIR"
echo "POLICY_GATE_AUDIT_JSON=$LATEST_JSON"
echo "POLICY_GATE_AUDIT_MD=$LATEST_MD"
