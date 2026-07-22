#!/usr/bin/env bash
# Materialize train-only false signal-block contexts for manual inspection.
#SBATCH --job-name=bib_signal_review
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
INPUT=${INPUT:?set train/validation source JSONL}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set frozen train-OOF line model}
SIGNAL_TCN_DIR=${SIGNAL_TCN_DIR:?set frozen train-OOF signal TCN}
SIGNAL_BLOCK_DIR=${SIGNAL_BLOCK_DIR:?set frozen train-OOF signal blocks}
QUALITY_DECISIONS=${QUALITY_DECISIONS:?set reviewed train-quality decisions}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable review directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_signal_block_review \
  --input "$INPUT" \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --signal-tcn-dir "$SIGNAL_TCN_DIR" \
  --signal-block-dir "$SIGNAL_BLOCK_DIR" \
  --quality-decisions "$QUALITY_DECISIONS" \
  --output-dir "$OUTPUT_DIR" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_SIGNAL_REVIEW=$OUTPUT_DIR"
