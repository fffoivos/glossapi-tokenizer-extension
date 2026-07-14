#!/usr/bin/env bash
# Materialize target-aligned sparse 32-bin position maps on Clariden CPU.
#SBATCH --job-name=bib_sparse_pos
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned scipy/sklearn target directory}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen entry table}
POSITIONAL_TABLE_DIR=${POSITIONAL_TABLE_DIR:?set lossless positional table}
ROLE_TARGET_DIR=${ROLE_TARGET_DIR:?set reviewed role target view}
OUTPUT_DIR=${OUTPUT_DIR:?set new sparse table output}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_sparse_position_table \
    --base-table-dir "$BASE_TABLE_DIR" --positional-table-dir "$POSITIONAL_TABLE_DIR" \
    --role-target-dir "$ROLE_TARGET_DIR" --output-dir "$OUTPUT_DIR" \
    --bins 32 --chunk-lines 50000 --workers 16 \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_SPARSE_POSITION=$OUTPUT_DIR"
