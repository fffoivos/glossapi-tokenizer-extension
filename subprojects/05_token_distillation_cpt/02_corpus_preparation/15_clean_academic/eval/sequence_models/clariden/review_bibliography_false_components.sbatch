#!/usr/bin/env bash
# Compact text packet for the largest train-OOF false components.
#SBATCH --job-name=bib_false_review
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
INPUT=${INPUT:?set physically test-stripped STRUCT-2K JSONL}
TABLE_DIR=${TABLE_DIR:?set train-only table}
COMPONENT_DIR=${COMPONENT_DIR:?set train-OOF component outputs}
OUTPUT_PATH=${OUTPUT_PATH:?set new review packet path}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 90
[[ -f "$TABLE_DIR/receipt.json" && ! -L "$TABLE_DIR/receipt.json" ]] || exit 91
[[ -f "$COMPONENT_DIR/receipt.json" && ! -L "$COMPONENT_DIR/receipt.json" ]] || exit 92
[[ ! -e "$OUTPUT_PATH" && ! -L "$OUTPUT_PATH" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_component_false_review \
  --input "$INPUT" \
  --table-dir "$TABLE_DIR" \
  --component-dir "$COMPONENT_DIR" \
  --output-path "$OUTPUT_PATH"

echo "BIBLIOGRAPHY_FALSE_COMPONENT_REVIEW=$OUTPUT_PATH"
