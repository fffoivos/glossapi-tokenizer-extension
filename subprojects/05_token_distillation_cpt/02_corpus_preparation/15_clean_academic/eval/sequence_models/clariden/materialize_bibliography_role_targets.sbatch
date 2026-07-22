#!/usr/bin/env bash
# Join reviewed role truth to the frozen entry table on Clariden CPU.
#SBATCH --job-name=bib_role_targets
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
INPUT=${INPUT:?set STRUCT-2K source}
EXPECTED_INPUT_SHA256=${EXPECTED_INPUT_SHA256:?pin source}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen entry table}
OVERLAY=${OVERLAY:?set final role overlay v2}
OUTPUT_DIR=${OUTPUT_DIR:?set new target view directory}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_role_target_view \
    --input "$INPUT" --expected-input-sha256 "$EXPECTED_INPUT_SHA256" \
    --base-table-dir "$BASE_TABLE_DIR" --overlay "$OVERLAY" \
    --output-dir "$OUTPUT_DIR" --split train --code-commit "$EXPECTED_COMMIT" \
    --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_ROLE_TARGETS=$OUTPUT_DIR"
