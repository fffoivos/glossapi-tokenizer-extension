#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-gate
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SEGMENT_ID:?set segment ID}"
: "${RUN_ROOT:?set run root}"
: "${CHECKPOINT_RECEIPT:?set checkpoint receipt}"
: "${SEGMENT_GATE_RECEIPT:?set gate receipt output}"
exec "$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/gate_segment.py" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" --segment-id "$SEGMENT_ID" \
  --run-root "$RUN_ROOT" --checkpoint-receipt "$CHECKPOINT_RECEIPT" \
  --output "$SEGMENT_GATE_RECEIPT"
