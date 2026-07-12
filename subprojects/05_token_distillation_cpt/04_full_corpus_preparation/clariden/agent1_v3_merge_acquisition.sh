#!/usr/bin/env bash
# Merge fresh v3-code HF and MDC receipts without copying source payloads.
set -euo pipefail

hf="$AGENT1_V3_PHASE0_DIR/hf_acquisition_receipt.json"
mdc="$AGENT1_V3_PHASE0_DIR/mdc_acquisition_receipt.json"
test -s "$hf" && test -s "$mdc"
output="$AGENT1_V3_ATTEMPT_DIR/merged.receipt.json"
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/merge_acquisition_receipts.py" \
    --sources "$AGENT1_V3_SOURCE_CONFIG" --hf-receipt "$hf" --mdc-receipt "$mdc" \
    --destination-root "$AGENT1_V3_RAW_COMMON_ROOT" --expected-code-commit "$AGENT1_V3_EXPECTED_COMMIT" \
    --output "$output"
ln -s "attempts/$AGENT1_V3_ATTEMPT_ID/merged.receipt.json" "$AGENT1_V3_PHASE0_DIR/merged_acquisition_receipt.json"
echo "AGENT1_V3_MERGED_ACQUISITION_RECEIPT=$output"
