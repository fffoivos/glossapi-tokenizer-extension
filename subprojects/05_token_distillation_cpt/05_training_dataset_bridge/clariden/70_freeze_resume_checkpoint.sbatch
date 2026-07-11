#!/usr/bin/env bash
#SBATCH --job-name=cpt5_resume_receipt
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${BRIDGE_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
bridge_require_paths
: "${PROBE_OUTPUT_DIR:?set the immutable probe output directory}"
: "${COMPLETED_ITERATION:?set the completed segment iteration}"
test -s "$PROBE_OUTPUT_DIR/probe_plan.json"
test -s "$TRAINING_ASSETS_RECEIPT"
mkdir -p "$PROBE_OUTPUT_DIR/checkpoint_receipts"
bridge_python "$RESUME_CHECKPOINT_SCRIPT" \
    --checkpoint-dir "$PROBE_OUTPUT_DIR/checkpoints" \
    --expected-iteration "$COMPLETED_ITERATION" \
    --probe-plan "$PROBE_OUTPUT_DIR/probe_plan.json" \
    --training-assets-receipt "$TRAINING_ASSETS_RECEIPT" \
    --output "$PROBE_OUTPUT_DIR/checkpoint_receipts/iter_${COMPLETED_ITERATION}.json"
