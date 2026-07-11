#!/usr/bin/env bash
#SBATCH --job-name=cpt5_heldouts
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=420G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${BRIDGE_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
bridge_require_paths
test -s "$INPUT_RECEIPT"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bridge_python "$HELDOUT_SCRIPT" \
    --input-receipt "$INPUT_RECEIPT" \
    --config "$BRIDGE_CONFIG" \
    --stage-root "$BRIDGE_STAGE_ROOT" \
    --workers "${HELDOUT_WORKERS:-6}"
