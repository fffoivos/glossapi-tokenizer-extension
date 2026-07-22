#!/usr/bin/env bash
# Select a source/fold/shape-stratified heading review subset on Clariden CPU.
#SBATCH --job-name=bib_heading_select
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
INVENTORY_DIR=${INVENTORY_DIR:?set contextual heading inventory root}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable review-selection root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
PER_SOURCE_QUOTA=${PER_SOURCE_QUOTA:-500}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
mkdir "$OUTPUT_DIR"
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_heading_review select \
    --packet "$INVENTORY_DIR/packet.json" \
    --provenance "$INVENTORY_DIR/provenance.json" \
    --per-source-quota "$PER_SOURCE_QUOTA" \
    --packet-out "$OUTPUT_DIR/packet.json" \
    --provenance-out "$OUTPUT_DIR/provenance.json" \
    --receipt-out "$OUTPUT_DIR/receipt.json" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_HEADING_REVIEW_SELECTION=$OUTPUT_DIR"
