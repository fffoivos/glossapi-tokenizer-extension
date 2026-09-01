#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-supervisor
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${RUN_ROOT:?set campaign run root}"
: "${RUN_TAG:?set campaign run tag}"
: "${SEGMENT_ID:?set segment ID}"
: "${SEGMENT_ATTEMPT:?set segment attempt}"
: "${TRAIN_JOB_ID:?set aggregate train job ID}"
: "${WATCH_JOB_ID:?set evaluation watcher job ID}"

args=(
  --campaign-manifest "$CAMPAIGN_MANIFEST" --run-root "$RUN_ROOT"
  --run-tag "$RUN_TAG" --segment-id "$SEGMENT_ID"
  --segment-attempt "$SEGMENT_ATTEMPT" --train-job-id "$TRAIN_JOB_ID"
  --watch-job-id "$WATCH_JOB_ID"
)
if [[ -n "${RESUME_RECEIPT:-}" ]]; then args+=(--resume-receipt "$RESUME_RECEIPT"); fi
if [[ -n "${RECOVERY_START:-}" ]]; then args+=(--recovery-start "$RECOVERY_START"); fi
CONTROLLER_BUNDLE=${RECOVERY_CONTROLLER_BUNDLE:-$SCIENTIFIC_BUNDLE}
exec "$HOST_PYTHON" "$CONTROLLER_BUNDLE/production/supervise_production_segment.py" "${args[@]}"
