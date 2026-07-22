#!/usr/bin/env bash
# Train a small train-only signal TCN with document-grouped OOF predictions.
#SBATCH --job-name=bib_signal_tcn
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
INPUT=${INPUT:?set train/validation source JSONL; validation rows are rejected before parsing}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set frozen train-OOF line model}
BLOCK_OOF_DIR=${BLOCK_OOF_DIR:?set frozen train-OOF block config}
DETERMINISTIC_ROLES_DIR=${DETERMINISTIC_ROLES_DIR:?set train-only role arrays}
QUALITY_DECISIONS=${QUALITY_DECISIONS:?set reviewed train-quality decisions}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable signal-TCN directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

for receipt in \
  "$TABLE_DIR/receipt.json" \
  "$LINE_OOF_DIR/receipt.json" \
  "$BLOCK_OOF_DIR/receipt.json" \
  "$DETERMINISTIC_ROLES_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 90
done
[[ -f "$INPUT" && ! -L "$INPUT" ]] || exit 91
[[ -f "$QUALITY_DECISIONS" && ! -L "$QUALITY_DECISIONS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_signal_tcn \
  --input "$INPUT" \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --block-oof-dir "$BLOCK_OOF_DIR" \
  --deterministic-roles-dir "$DETERMINISTIC_ROLES_DIR" \
  --quality-decisions "$QUALITY_DECISIONS" \
  --output-dir "$OUTPUT_DIR" \
  --workers 5 \
  --cpus "$SLURM_CPUS_PER_TASK" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_SIGNAL_TCN=$OUTPUT_DIR"
