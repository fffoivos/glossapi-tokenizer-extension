#!/usr/bin/env bash
# Fit the six train-only OOF bibliography entry-line arms on Clariden CPU.
#SBATCH --job-name=bib_entry_oof
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
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable line-OOF directory}
CODE_COMMIT=${CODE_COMMIT:?set source commit}
UENV=${UENV:-pytorch/v2.9.1:v2}

[[ -d "$CODE_ROOT/sequence_models" && ! -L "$CODE_ROOT" ]] || exit 90
[[ -f "$TABLE_DIR/receipt.json" && ! -L "$TABLE_DIR" ]] || exit 91
[[ -d "$PY_DEPS" && -f "$PY_DEPS/sklearn/__init__.py" && ! -L "$PY_DEPS" ]] || exit 92
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 93

export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void
export ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONHASHSEED=0
uenv run "$UENV" --view=default -- \
  env PYTHONPATH="$PY_DEPS:$CODE_ROOT" python3 -m sequence_models.bibliography_entry_models \
  --table-dir "$TABLE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --c-grid 0.1 1.0 \
  --seed 1729 \
  --code-commit "$CODE_COMMIT" \
  --slurm-job-id "$SLURM_JOB_ID"

echo "BIBLIOGRAPHY_ENTRY_LINE_OOF=$OUTPUT_DIR"
