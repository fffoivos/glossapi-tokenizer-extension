#!/usr/bin/env bash
#SBATCH --job-name=cpt5_restage
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=24:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${BRIDGE_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
bridge_require_base
mkdir -p "$BRIDGE_STAGE_ROOT" "$BRIDGE_RUN_ROOT" "$REPLAY_HF_CACHE"
: "${HF_TOKEN:?HF_TOKEN must be injected into the restaging job environment}"
replace=()
[[ "${RESTAGE_REPLACE:-0}" == 0 ]] || replace=(--replace-existing)
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=120
bridge_python "$ACQUIRE_SCRIPT" \
    --config "$REPLAY_ACQUISITION_CONFIG" \
    --scratch-root "$SCRATCH_ROOT" \
    --cache-dir "$REPLAY_HF_CACHE" \
    --output "$REPLAY_ACQUISITION_RECEIPT" \
    "${replace[@]}"
