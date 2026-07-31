#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_final
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
set -euo pipefail
HERE=${CPT_CLARIDEN_DIR:?submit through submit_data_pipeline.sh}
source "$HERE/paths.env"
cpt_require_runtime
test -s "$INPUT_RECEIPT"
test -s "$HELDOUT_MANIFEST"
cpt_python "$CPT_DIR/dataset/finalize_phase_bridge.py" \
  --recipe "$RECIPE" --input-receipt "$INPUT_RECEIPT" \
  --heldout-manifest "$HELDOUT_MANIFEST" --stage-root "$STAGE_ROOT"
