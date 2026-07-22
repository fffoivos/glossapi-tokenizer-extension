#!/usr/bin/env bash
# Split frozen signal-TCN blocks at train-only deterministic barriers.
#SBATCH --job-name=bib_signal_barriers
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root}
INPUT=${INPUT:?set train/validation source JSONL}
TABLE_DIR=${TABLE_DIR:?set train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set frozen train-OOF line model}
SIGNAL_TCN_DIR=${SIGNAL_TCN_DIR:?set frozen train-OOF signal TCN}
DETERMINISTIC_ROLES_DIR=${DETERMINISTIC_ROLES_DIR:?set frozen train-only roles}
QUALITY_DECISIONS=${QUALITY_DECISIONS:?set reviewed extraction-quality decisions}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable barrier-decode directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

for receipt in "$TABLE_DIR/receipt.json" "$LINE_OOF_DIR/receipt.json" "$SIGNAL_TCN_DIR/receipt.json" "$DETERMINISTIC_ROLES_DIR/receipt.json"; do
  [[ -f "$receipt" && ! -L "$receipt" ]] || exit 90
done
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$CODE_ROOT" python3 -m sequence_models.bibliography_signal_barrier_decode \
  --input "$INPUT" \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --signal-tcn-dir "$SIGNAL_TCN_DIR" \
  --deterministic-roles-dir "$DETERMINISTIC_ROLES_DIR" \
  --quality-decisions "$QUALITY_DECISIONS" \
  --output-dir "$OUTPUT_DIR" \
  --workers "$SLURM_CPUS_PER_TASK" \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_SIGNAL_BARRIERS=$OUTPUT_DIR"
