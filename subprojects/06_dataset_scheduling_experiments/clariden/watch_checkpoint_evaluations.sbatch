#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-eval-watch
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SEGMENT_ID:?set segment ID}"
: "${RUN_ROOT:?set campaign run root}"
: "${TRAIN_JOB_ID:?set aggregate training job ID}"
SEGMENT_ATTEMPT=${SEGMENT_ATTEMPT:-0}
watch_args=(
  --campaign-manifest "$CAMPAIGN_MANIFEST" --segment-id "$SEGMENT_ID"
  --segment-attempt "$SEGMENT_ATTEMPT" --run-root "$RUN_ROOT"
  --train-job-id "$TRAIN_JOB_ID"
)
if [[ -n "${RECOVERY_START:-}" ]]; then watch_args+=(--recovery-start "$RECOVERY_START"); fi
if [[ -n "${RECOVERY_RECEIPT:-}" ]]; then watch_args+=(--recovery-receipt "$RECOVERY_RECEIPT"); fi
exec "$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/watch_checkpoint_evaluations.py" \
  "${watch_args[@]}"
