#!/usr/bin/env bash
# Freeze the downstream v3 contract only after the merged acquisition receipt.
set -euo pipefail

merged="$AGENT1_V3_PHASE0_DIR/merged_acquisition_receipt.json"
test -s "$merged"
"$AGENT1_V3_RUNTIME_VENV/bin/python" "$AGENT1_V3_CONTRACT_SCRIPT" freeze-run \
    --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID" \
    --code-commit "$AGENT1_V3_EXPECTED_COMMIT" \
    --source-registry "$AGENT1_V3_SOURCE_CONFIG" \
    --source-aliases "$AGENT1_V3_SOURCE_ALIASES" \
    --candidate-roster "$AGENT1_V3_CANDIDATE_ROSTER" \
    --acquisition-receipt "$merged" \
    --tokenizer "$AGENT1_V3_TOKENIZER_JSON" \
    --review-policy "$AGENT1_V3_REVIEW_POLICY" \
    --review-prompt "$AGENT1_V3_REVIEW_PROMPT" \
    --review-response-schema "$AGENT1_V3_REVIEW_RESPONSE_SCHEMA" \
    --dedup-policy "$AGENT1_V3_POLICY" \
    --greekmmlu-policy "$AGENT1_V3_POLICY" \
    --anonymization-policy "$AGENT1_V3_POLICY" \
    --structural-policy "$AGENT1_V3_POLICY" \
    --prestructural-only
