#!/usr/bin/env bash
# Fit grouped OOF heading or connector experts on Clariden CPU.
#SBATCH --job-name=bib_role_expert
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
PY_DEPS=${PY_DEPS:?set pinned scikit-learn dependency root}
KIND=${KIND:?set heading or connector}
TABLE_DIR=${TABLE_DIR:?set role expert table}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable model root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ "$KIND" == heading || "$KIND" == connector ]] || exit 91
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT:$PY_DEPS" \
  python3 -m sequence_models.bibliography_role_experts \
    --kind "$KIND" --table-dir "$TABLE_DIR" --output-dir "$OUTPUT_DIR" \
    --seed 20260715 --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_ROLE_EXPERT=$OUTPUT_DIR"
