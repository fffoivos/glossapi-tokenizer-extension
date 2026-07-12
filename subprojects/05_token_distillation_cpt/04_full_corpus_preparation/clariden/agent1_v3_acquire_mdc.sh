#!/usr/bin/env bash
# Receipt-bound Mozilla Data Collective acquisition into the v3 raw namespace.
set -euo pipefail

[[ -n "${MOZILLA_DATA_COLLECTIVE_API_KEY:-}" ]] || {
    echo "ERROR: inject MOZILLA_DATA_COLLECTIVE_API_KEY only for this submission" >&2
    exit 31
}
destination="$AGENT1_V3_DATA_ROOT/raw/mdc"
receipt="$AGENT1_V3_ATTEMPT_DIR/mdc.receipt.json"
mkdir -p "$destination"
command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 31; }
exec 9>"$destination/.agent1-v3-mdc.lock"
flock -n 9 || { echo "ERROR: another v3 MDC acquisition is active" >&2; exit 31; }

uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/acquire_mdc_sources.py" \
    --sources "$AGENT1_V3_SOURCE_CONFIG" --destination "$destination" --output "$receipt" \
    --code-commit "$AGENT1_V3_EXPECTED_COMMIT"
ln -s "attempts/$AGENT1_V3_ATTEMPT_ID/mdc.receipt.json" "$AGENT1_V3_PHASE0_DIR/mdc_acquisition_receipt.json"
unset MOZILLA_DATA_COLLECTIVE_API_KEY
echo "AGENT1_V3_MDC_ACQUISITION_RECEIPT=$receipt"
