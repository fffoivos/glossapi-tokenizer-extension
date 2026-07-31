#!/usr/bin/env bash
#SBATCH --job-name=cpt25b_replay
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
: "${HF_TOKEN:?inject HF_TOKEN}"
mkdir -p "$STAGE_ROOT/receipts" "$RUN_ROOT" "$SCRATCH_ROOT/hf_cache/cpt25b_replay"
replace=()
[[ "${RESTAGE_REPLACE:-0}" == 0 ]] || replace=(--replace-existing)
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=120
cpt_python "$LEGACY_DIR/scripts/acquire_replay_sources.py" \
  --config "$REPLAY_CONFIG" --scratch-root "$SCRATCH_ROOT" \
  --cache-dir "$SCRATCH_ROOT/hf_cache/cpt25b_replay" --output "$REPLAY_RECEIPT" \
  "${replace[@]}"
