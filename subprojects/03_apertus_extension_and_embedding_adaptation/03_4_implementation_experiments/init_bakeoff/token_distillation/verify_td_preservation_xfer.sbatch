#!/usr/bin/env bash
#SBATCH --job-name=td_preserve_check
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/token_distillation/%x-%j.err

set -euo pipefail

REFERENCE_HF_DIR="${REFERENCE_HF_DIR:-/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/retok}"
TD_HF_DIR="${TD_HF_DIR:?TD_HF_DIR is required}"
MANIFEST="${MANIFEST:-$TD_HF_DIR/retok_td_manifest.json}"
OUTPUT_JSON="${OUTPUT_JSON:-$TD_HF_DIR/td_preservation_report.json}"
PYTHON="${PYTHON:-python3}"

if [ -n "${SCRIPT_DIR_OVERRIDE:-}" ]; then
    SCRIPT_DIR="$(cd "$SCRIPT_DIR_OVERRIDE" && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/verify_td_preservation.py" ]; then
    SCRIPT_DIR="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

echo "=== verify_td_preservation_xfer.sbatch ==="
date -u
echo "host: $HOSTNAME"
echo "REFERENCE_HF_DIR: $REFERENCE_HF_DIR"
echo "TD_HF_DIR:        $TD_HF_DIR"
echo "MANIFEST:         $MANIFEST"
echo "OUTPUT_JSON:      $OUTPUT_JSON"
echo "PYTHON:           $PYTHON"

"$PYTHON" "$SCRIPT_DIR/verify_td_preservation.py" \
  --reference-hf-dir "$REFERENCE_HF_DIR" \
  --td-hf-dir "$TD_HF_DIR" \
  --manifest "$MANIFEST" \
  --output-json "$OUTPUT_JSON"
