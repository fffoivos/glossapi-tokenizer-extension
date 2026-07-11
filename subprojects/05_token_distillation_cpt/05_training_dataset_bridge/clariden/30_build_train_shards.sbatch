#!/usr/bin/env bash
#SBATCH --job-name=cpt5_train_bins
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%A_%a.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%A_%a.err
set -euo pipefail

HERE=${BRIDGE_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
bridge_require_paths
test -s "$INPUT_RECEIPT"
test -s "$HELDOUT_MANIFEST"
TASK_INDEX=${SLURM_ARRAY_TASK_ID:?this stage must be a Slurm array}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bridge_python "$BINARY_SCRIPT" \
    --input-receipt "$INPUT_RECEIPT" \
    --heldout-manifest "$HELDOUT_MANIFEST" \
    --stage-root "$BRIDGE_STAGE_ROOT" \
    --task-index "$TASK_INDEX" \
    --workers "${BINARY_WORKERS:-48}" \
    --chunksize "${BINARY_CHUNKSIZE:-8}"
