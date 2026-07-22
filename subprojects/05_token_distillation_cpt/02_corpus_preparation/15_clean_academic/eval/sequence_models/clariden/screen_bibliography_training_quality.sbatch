#!/usr/bin/env bash
# Prediction- and label-blind extraction-quality screen for train documents.
#SBATCH --job-name=bib_train_quality
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
NOISE_DEPS=${NOISE_DEPS:?set directory containing glossapi_rs_noise wheel install}
INPUT=${INPUT:?set physically test-stripped STRUCT-2K JSONL}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable train-quality directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 90
[[ -f "$TABLE_DIR/receipt.json" && ! -L "$TABLE_DIR" ]] || exit 91
[[ -d "$NOISE_DEPS" && ! -L "$NOISE_DEPS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$NOISE_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_training_quality \
  --input "$INPUT" \
  --table-dir "$TABLE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_TRAINING_QUALITY=$OUTPUT_DIR"
