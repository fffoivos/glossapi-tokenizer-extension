#!/usr/bin/env bash
#SBATCH --job-name=full8b-docval-debug
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_VALIDATION_MANIFEST:?set frozen validation manifest}"
: "${FULL8_HF_MODEL:?set exported HF checkpoint}"
: "${FULL8_HF_TOKENIZER:?set frozen tokenizer}"
: "${FULL8_DOCVAL_OUTPUT:?set checkpoint document output root}"
: "${SLURM_ARRAY_TASK_ID:?set document group 0 through 3}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "per-document validation must run on debug" >&2
  exit 2
}
[[ "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "per-document validation must use exactly one node per group" >&2
  exit 2
}
exec bash "$FULL8_OPS_ROOT/clariden/run_per_document_group_resource_aware.sh"
