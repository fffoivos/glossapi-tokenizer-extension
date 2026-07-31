#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_heldout
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=420G
#SBATCH --time=12:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
cpt_require_runtime
test -s "$INPUT_RECEIPT"
cpt_python "$LEGACY_DIR/scripts/build_heldouts.py" \
  --input-receipt "$INPUT_RECEIPT" --config "$RECIPE" --stage-root "$STAGE_ROOT" \
  --workers "${HELDOUT_WORKERS:-6}"
