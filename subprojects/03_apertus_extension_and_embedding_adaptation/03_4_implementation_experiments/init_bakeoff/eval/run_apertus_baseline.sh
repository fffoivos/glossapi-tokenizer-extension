#!/usr/bin/env bash
# V4 baseline: full eval suite on unmodified Apertus-8B-2509.
# Output gates the §5.6 hard-gate thresholds for the bakeoff arms.
#
# Thin wrapper around run_eval.sbatch; sets MODEL_PATH to the staged
# Apertus base and TASK_GROUP=full, then submits via sbatch.
#
# Usage:
#   bash run_apertus_baseline.sh

set -euo pipefail

APERTUS_BASE="${APERTUS_BASE:-/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509}"
OUT_ROOT="${OUT_ROOT:-/capstor/scratch/cscs/fffoivos/runs/eval}"
RUN_NAME="apertus_baseline_v4_$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$OUT_ROOT/$RUN_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Submitting V4 baseline eval"
echo "  MODEL_PATH=$APERTUS_BASE"
echo "  OUTPUT_DIR=$OUTPUT_DIR"

sbatch \
    --export=ALL,MODEL_PATH="$APERTUS_BASE",OUTPUT_DIR="$OUTPUT_DIR",TASK_GROUP=full \
    --job-name="eval_baseline" \
    "$(dirname "$0")/run_eval.sbatch"

echo
echo "When done, run:"
echo "  python3 compute_bootstrap_cis.py $OUTPUT_DIR/samples_*.jsonl > $OUTPUT_DIR/bootstrap_cis.json"
