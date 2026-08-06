#!/usr/bin/env bash
#SBATCH --job-name=full8b-supervisor
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable bundle receipt}"
: "${FULL8_STAGE_ROOT:?set stage root}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_INITIAL_MEGATRON:?set initial checkpoint}"
: "${FULL8_SELECTED_PROFILE:?set selected profile receipt}"
: "${FULL8_LAUNCH_GATE:?set launch gate receipt}"
: "${FULL8_PRELAUNCH_ROOT:?set completed prelaunch root}"
[[ "$#" -eq 4 ]] || { echo "usage: $0 SEGMENT ATTEMPT ATTEMPT_START TRAIN_JOB" >&2; exit 2; }
command=(uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/supervise_campaign.py" \
  --code-root "$FULL8_CODE_ROOT" --stage-root "$FULL8_STAGE_ROOT" --run-root "$FULL8_RUN_ROOT" \
  --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --initial-megatron "$FULL8_INITIAL_MEGATRON" --selected-profile "$FULL8_SELECTED_PROFILE" \
  --launch-gate "$FULL8_LAUNCH_GATE" --prelaunch-root "$FULL8_PRELAUNCH_ROOT" \
  --segment-id "$1" --attempt "$2" --attempt-start "$3" --train-job-id "$4")
if [[ -n "${FULL8_PREQUEUED_MANIFEST:-}" ]]; then
  command+=(--prequeued-manifest "$FULL8_PREQUEUED_MANIFEST")
fi
exec "${command[@]}"
