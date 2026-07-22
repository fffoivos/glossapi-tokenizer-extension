#!/usr/bin/env bash
# Train constrained B1 CRFs from retained train-only OOF line arms.
#SBATCH --job-name=bib_entry_b1
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077

CODE_ROOT=${CODE_ROOT:?set immutable code-bundle root containing sequence_models}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn target directory}
TABLE_DIR=${TABLE_DIR:?set completed train-only feature table}
LINE_OOF_DIR=${LINE_OOF_DIR:?set completed line-OOF directory}
BLOCK_OOF_DIR=${BLOCK_OOF_DIR:?set completed B0/H0 directory}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable B1 directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -d "$CODE_ROOT/sequence_models" && ! -L "$CODE_ROOT" ]] || exit 90
[[ -f "$TABLE_DIR/receipt.json" && ! -L "$TABLE_DIR" ]] || exit 91
[[ -f "$LINE_OOF_DIR/receipt.json" && ! -L "$LINE_OOF_DIR" ]] || exit 92
[[ -f "$BLOCK_OOF_DIR/receipt.json" && ! -L "$BLOCK_OOF_DIR" ]] || exit 93
[[ -d "$PY_DEPS" && -f "$PY_DEPS/sklearn/__init__.py" && ! -L "$PY_DEPS" ]] || exit 94
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 95

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0

uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_entry_sequence \
  --table-dir "$TABLE_DIR" \
  --line-oof-dir "$LINE_OOF_DIR" \
  --block-oof-dir "$BLOCK_OOF_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 3 \
  --learning-rate 0.08 \
  --l2 0.0001 \
  --gradient-clip 5.0 \
  --workers 20 \
  --seed 2718 \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ENTRY_B1=$OUTPUT_DIR"
