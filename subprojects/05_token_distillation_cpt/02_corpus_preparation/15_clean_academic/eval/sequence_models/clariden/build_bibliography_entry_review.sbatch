#!/usr/bin/env bash
# Build a source-balanced, complete-context review packet after validation.
#SBATCH --job-name=bib_entry_review
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn target directory}
INPUT=${INPUT:?set physically test-stripped STRUCT-2K JSONL}
VALIDATION_DIR=${VALIDATION_DIR:?set completed retrospective-validation directory}
FREEZE_DIR=${FREEZE_DIR:?set completed frozen selection directory}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable review-site directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 90
[[ -f "$VALIDATION_DIR/receipt.json" && ! -L "$VALIDATION_DIR" ]] || exit 91
[[ -f "$FREEZE_DIR/receipt.json" && ! -L "$FREEZE_DIR" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_entry_review \
  --input "$INPUT" \
  --validation-dir "$VALIDATION_DIR" \
  --frozen-config "$FREEZE_DIR/frozen_config.json" \
  --output-dir "$OUTPUT_DIR" \
  --case-count 120 \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ENTRY_REVIEW=$OUTPUT_DIR"
