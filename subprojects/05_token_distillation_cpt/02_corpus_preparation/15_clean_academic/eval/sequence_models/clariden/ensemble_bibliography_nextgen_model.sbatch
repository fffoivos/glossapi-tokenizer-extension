#!/usr/bin/env bash
#SBATCH --job-name=bib-ng-mix
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
MODEL_A=${MODEL_A:?set first model root}
MODEL_B=${MODEL_B:?set second model root}
WEIGHT_A=${WEIGHT_A:?set first model weight}
WEIGHT_B=${WEIGHT_B:?set second model weight}
ENSEMBLE_NAME=${ENSEMBLE_NAME:?set ensemble name}
OUTPUT_DIR=${OUTPUT_DIR:?set new immutable ensemble root}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_nextgen_ensemble \
    --model-dirs "$MODEL_A" "$MODEL_B" --weights "$WEIGHT_A" "$WEIGHT_B" \
    --name "$ENSEMBLE_NAME" --output-dir "$OUTPUT_DIR" \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_NEXTGEN_ENSEMBLE=$OUTPUT_DIR"
