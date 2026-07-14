#!/usr/bin/env bash
# Prepare additional role-review blocks and zero-BIB controls on Clariden CPU.
#SBATCH --job-name=bib_role_bootstrap
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code root}
INPUT=${INPUT:?set STRUCT-2K input}
EXPECTED_INPUT_SHA256=${EXPECTED_INPUT_SHA256:?pin STRUCT-2K input}
PROFILE_ROOT=${PROFILE_ROOT:?set authoritative role profile root}
OUTPUT_DIR=${OUTPUT_DIR:?set new bootstrap output}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT_DIR" && ! -L "$OUTPUT_DIR" ]] || exit 90
trap 'rm -rf "$OUTPUT_DIR"' EXIT
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_role_bootstrap \
    --input "$INPUT" --expected-input-sha256 "$EXPECTED_INPUT_SHA256" \
    --block-inventory "$PROFILE_ROOT/block_inventory.jsonl" \
    --initial-selection "$PROFILE_ROOT/selection_manifest.provenance.json" \
    --output-dir "$OUTPUT_DIR" --per-source 80 --workers "$SLURM_CPUS_PER_TASK" \
    --seed bibliography-role-bootstrap-v1 --code-commit "$EXPECTED_COMMIT" \
    --slurm-job-id "$SLURM_JOB_ID"
trap - EXIT
echo "BIBLIOGRAPHY_ROLE_BOOTSTRAP=$OUTPUT_DIR"
