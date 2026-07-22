#!/usr/bin/env bash
# Fit sparse 8/16/32-bin P2 arms and gate them against P0D.
#SBATCH --job-name=bib_p2_sparse
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
PY_DEPS=${PY_DEPS:?set pinned scipy/sklearn target directory}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen entry table}
ROLE_TARGET_DIR=${ROLE_TARGET_DIR:?set reviewed targets}
SPARSE_TABLE_DIR=${SPARSE_TABLE_DIR:?set verified sparse position table}
BASELINE_MODEL_DIR=${BASELINE_MODEL_DIR:?set completed P0D ladder}
OUTPUT_DIR=${OUTPUT_DIR:?set new P2 output}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_sparse_position_models \
    --base-table-dir "$BASE_TABLE_DIR" --role-target-dir "$ROLE_TARGET_DIR" \
    --sparse-table-dir "$SPARSE_TABLE_DIR" --baseline-model-dir "$BASELINE_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" --alpha-grid 0.000001 0.000003 0.00001 0.00003 \
    --precision-target 0.99 --maximum-source-recall-loss 0.01 \
    --bootstrap-replicates 2000 --parallel-folds 5 --seed 20260714 \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_P2_SPARSE=$OUTPUT_DIR"
