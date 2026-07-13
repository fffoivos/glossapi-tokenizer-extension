#!/usr/bin/env bash
# Freeze the downstream v3 contract only after the merged acquisition receipt.
set -euo pipefail

merged="$AGENT1_V3_PHASE0_DIR/merged_acquisition_receipt.json"
test -s "$merged"
test -s "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT"
test -s "$AGENT1_V3_POST_CUTOFF_INVENTORY"
test -s "$AGENT1_V3_NANOCHAT_INITIAL_ROSTER"
test -s "$AGENT1_V3_LICENSE_ADJUDICATION"
test -s "$AGENT1_V3_ELIGIBILITY_POLICY"
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$AGENT1_V3_CONTRACT_SCRIPT" freeze-run \
    --run-root "$AGENT1_V3_RUN_ROOT" --run-id "$AGENT1_V3_RUN_ID" \
    --code-commit "$AGENT1_V3_EXPECTED_COMMIT" \
    --source-registry "$AGENT1_V3_SOURCE_CONFIG" \
    --source-aliases "$AGENT1_V3_SOURCE_ALIASES" \
    --candidate-roster "$AGENT1_V3_CANDIDATE_ROSTER" \
    --post-cutoff-inventory "$AGENT1_V3_POST_CUTOFF_INVENTORY" \
    --nanochat-initial-roster "$AGENT1_V3_NANOCHAT_INITIAL_ROSTER" \
    --acquisition-receipt "$merged" \
    --tokenizer "$AGENT1_V3_TOKENIZER_JSON" \
    --review-policy "$AGENT1_V3_REVIEW_POLICY" \
    --review-prompt "$AGENT1_V3_REVIEW_PROMPT" \
    --review-response-schema "$AGENT1_V3_REVIEW_RESPONSE_SCHEMA" \
    --glossapi-build-receipt "$AGENT1_V3_GLOSSAPI_BUILD_RECEIPT" \
    --license-adjudication "$AGENT1_V3_LICENSE_ADJUDICATION" \
    --training-eligibility-policy "$AGENT1_V3_ELIGIBILITY_POLICY" \
    --dedup-policy "$AGENT1_V3_POLICY" \
    --greekmmlu-policy "$AGENT1_V3_POLICY" \
    --anonymization-policy "$AGENT1_V3_POLICY" \
    --structural-policy "$AGENT1_V3_POLICY" \
    --prestructural-only
