#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-freeze
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SEGMENT_ID:?set segment ID}"
: "${RUN_ROOT:?set run root}"
: "${CHECKPOINT_RECEIPT:?set output receipt}"
SEGMENT_ATTEMPT=${SEGMENT_ATTEMPT:-0}
freeze_args=(
  --campaign-manifest "$CAMPAIGN_MANIFEST" --segment-id "$SEGMENT_ID"
  --segment-attempt "$SEGMENT_ATTEMPT" --run-root "$RUN_ROOT"
  --output "$CHECKPOINT_RECEIPT"
)
if [[ -n "${CHECKPOINT_ITERATION:-}" ]]; then freeze_args+=(--iteration "$CHECKPOINT_ITERATION"); fi
if [[ "${FAILED_ATTEMPT_RECOVERY:-0}" == "1" ]]; then freeze_args+=(--failed-attempt-recovery); fi
if [[ "${NUMERICAL_FAILURE_RECOVERY:-0}" == "1" ]]; then freeze_args+=(--numerical-failure-recovery); fi
if [[ -n "${LOAD_VIEW_ROOT:-}" ]]; then freeze_args+=(--load-view-root "$LOAD_VIEW_ROOT"); fi
exec "$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/freeze_segment_checkpoint.py" \
  "${freeze_args[@]}"
