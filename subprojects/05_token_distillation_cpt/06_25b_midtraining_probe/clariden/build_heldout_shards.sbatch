#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_valbins
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=12:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
cpt_require_runtime
test -s "$INPUT_RECEIPT"
test -s "$HELDOUT_MANIFEST"
INDEX=${SLURM_ARRAY_TASK_ID:?array required}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cpt_python "$LEGACY_DIR/scripts/build_binary_shard.py" \
  --input-receipt "$INPUT_RECEIPT" --heldout-manifest "$HELDOUT_MANIFEST" \
  --stage-root "$STAGE_ROOT" --heldout-index "$INDEX" \
  --workers "${BINARY_WORKERS:-48}" --chunksize "${BINARY_CHUNKSIZE:-8}"
