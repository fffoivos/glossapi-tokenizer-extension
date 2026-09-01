#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-complete
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${RUN_ROOT:?set campaign run root}"
greek_args=()
if [[ -n "${FULL_ENDPOINT_VALIDATION_RECEIPT:-}" ]]; then
  greek_args+=(--full-endpoint-validation-receipt "$FULL_ENDPOINT_VALIDATION_RECEIPT")
fi
if [[ -n "${GREEK_ENDPOINT_RECEIPT:-}" ]]; then
  greek_args+=(--greek-endpoint-receipt "$GREEK_ENDPOINT_RECEIPT")
fi
if [[ -n "${RETENTION_ENDPOINT_RECEIPT:-}" ]]; then
  greek_args+=(--retention-endpoint-receipt "$RETENTION_ENDPOINT_RECEIPT")
fi
exec "$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/finalize_campaign_evidence.py" \
  --run-root "$RUN_ROOT" "${greek_args[@]}" \
  --output "$RUN_ROOT/campaign_evidence_completion_receipt.json"
