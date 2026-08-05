#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-finalize-core
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${RUN_ROOT:?set campaign run root}"
out="$RUN_ROOT/analysis"
mkdir -p "$out"
FULL_ENDPOINT_VALIDATION_RECEIPT=${FULL_ENDPOINT_VALIDATION_RECEIPT:-$RUN_ROOT/full_endpoint_validation/full_endpoint_validation_receipt.json}
validation_args=(
  --campaign-manifest "$CAMPAIGN_MANIFEST" --run-root "$RUN_ROOT" \
  --initial-validation-receipt "$RUN_ROOT/initial_validation/initial_validation_receipt.json" \
  --output-json "$out/validation_trajectory.json" --output-csv "$out/validation_trajectory.csv"
)
if [[ -n "${SOURCE_VALIDATION_ATTEMPT_AUTHORITY:-}" ]]; then
  IFS=';' read -r -a authority_items <<<"$SOURCE_VALIDATION_ATTEMPT_AUTHORITY"
  for item in "${authority_items[@]}"; do
    [[ -n "$item" ]] && validation_args+=(--attempt-authority-through "$item")
  done
fi
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/collect_validation_trajectory.py" \
  "${validation_args[@]}"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/collect_greekmmlu_trajectory.py" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" --run-root "$RUN_ROOT" \
  --output-json "$out/greekmmlu_trajectory.json" --output-csv "$out/greekmmlu_trajectory.csv"
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/evaluation/summarize_core_campaign.py" \
  --validation-trajectory "$out/validation_trajectory.json" \
  --greekmmlu-trajectory "$out/greekmmlu_trajectory.json" \
  --full-endpoint-validation "$FULL_ENDPOINT_VALIDATION_RECEIPT" \
  --output "$out/core_campaign_summary.json"
