#!/usr/bin/env bash
#SBATCH --job-name=cpt5_old_greek
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${BRIDGE_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
bridge_require_base
test -s "$REPLAY_ACQUISITION_RECEIPT"
replace=()
[[ "${RESTAGE_REPLACE:-0}" == 0 ]] || replace=(--replace-existing)
bridge_python "$OLD_GREEK_BUILD_SCRIPT" \
    --config "$REPLAY_ACQUISITION_CONFIG" \
    --acquisition-receipt "$REPLAY_ACQUISITION_RECEIPT" \
    --scratch-root "$SCRATCH_ROOT" \
    --output-receipt "$OLD_GREEK_BUILD_RECEIPT" \
    "${replace[@]}"
