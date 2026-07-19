#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-tcn
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
TABLE_DIR=${TABLE_DIR:?set nextgen feature table}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set entry feature table}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable model root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
MAXIMUM_POSITIVE_WEIGHT=${MAXIMUM_POSITIVE_WEIGHT:-1}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_nextgen_tcn \
    --table-dir "$TABLE_DIR" --base-table-dir "$BASE_TABLE_DIR" \
    --output-dir "$OUTPUT_DIR" --cpus "$SLURM_CPUS_PER_TASK" \
    --maximum-positive-weight "$MAXIMUM_POSITIVE_WEIGHT" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_NEXTGEN_TCN=$OUTPUT_DIR"
