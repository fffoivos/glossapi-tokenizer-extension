#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_oldgr
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
test -s "$REPLAY_RECEIPT"
cpt_python "$LEGACY_DIR/scripts/build_old_greek_replay.py" \
  --config "$REPLAY_CONFIG" --acquisition-receipt "$REPLAY_RECEIPT" \
  --scratch-root "$SCRATCH_ROOT" --output-receipt "$OLD_GREEK_RECEIPT"
