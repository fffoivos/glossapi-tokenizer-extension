#!/usr/bin/env bash
# Fit nested work-grouped corrected P0/P0D/P1/P1G on Clariden CPU.
#SBATCH --job-name=bib_pos_entry
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn target directory}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen entry table}
POSITIONAL_TABLE_DIR=${POSITIONAL_TABLE_DIR:?set positional feature table}
ROLE_TARGET_DIR=${ROLE_TARGET_DIR:?set reviewed role target view}
OUTPUT_DIR=${OUTPUT_DIR:?set new model output}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_positional_models \
    --base-table-dir "$BASE_TABLE_DIR" --positional-table-dir "$POSITIONAL_TABLE_DIR" \
    --role-target-dir "$ROLE_TARGET_DIR" --output-dir "$OUTPUT_DIR" \
    --c-grid 0.03 0.1 0.3 1.0 --seed 20260714 \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_POSITIONAL_ENTRY=$OUTPUT_DIR"
