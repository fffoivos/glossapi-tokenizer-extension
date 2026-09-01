#!/bin/bash
#SBATCH --job-name=gmmlu-history
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/logs/gmmlu-history-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/logs/gmmlu-history-%j.err

set -euo pipefail

: "${ANALYZER:?set ANALYZER to extract_greekmmlu_history_matrix.py}"
: "${DRIFT_RECEIPT:?set DRIFT_RECEIPT to greekmmlu_answer_drift.json}"
: "${OUTPUT:?set OUTPUT to the result NPZ path}"
: "${REPORT_BUILDER:?set REPORT_BUILDER to build_greekmmlu_historical_change_simulations.py}"
: "${REPORT_DATA:?set REPORT_DATA to the result data JSON path}"
: "${REPORT_HTML:?set REPORT_HTML to the result HTML path}"
UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}"

test "${SLURM_JOB_PARTITION:-}" = "debug"
test "${SLURM_JOB_NUM_NODES:-}" = "1"
mkdir -p "$(dirname "$OUTPUT")" "$(dirname "$REPORT_DATA")" "$(dirname "$REPORT_HTML")" /capstor/scratch/cscs/fffoivos/logs
uenv run "$UENV_IMAGE" --view=default -- \
  python3 "$ANALYZER" --drift-receipt "$DRIFT_RECEIPT" --output "$OUTPUT"
uenv run "$UENV_IMAGE" --view=default -- \
  python3 "$REPORT_BUILDER" \
    --drift "$DRIFT_RECEIPT" \
    --actual-history "$OUTPUT" \
    --output-data "$REPORT_DATA" \
    --output-html "$REPORT_HTML"
