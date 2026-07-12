#!/usr/bin/env bash
# Re-verify the existing HF payload on the exact v3 commit.  It must never
# download or copy the retained payload.
set -euo pipefail

[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: inject HF_TOKEN only for this submission" >&2; exit 30; }
attempt="$AGENT1_V3_ATTEMPT_DIR"
destination="$AGENT1_V3_HF_EXISTING_DESTINATION"
lock="$attempt/hf.lock.json"
download="$attempt/hf.download.json"
schemas="$attempt/hf.schemas.json"
receipt="$attempt/hf.receipt.json"

mkdir -p "$destination"
command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 30; }
exec 9>"$destination/.agent1-v3-existing-hf.lock"
flock -n 9 || { echo "ERROR: another existing-payload verification is active" >&2; exit 30; }
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60

uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/resolve_sources.py" \
    --sources "$AGENT1_V3_SOURCE_CONFIG" --output "$lock"
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/download_locked_sources.py" \
    --lock "$lock" --destination "$destination" --manifest "$download" --existing-only \
    --workers "${AGENT1_V3_HF_VERIFY_WORKERS:-12}" --download-attempts 1 --retry-backoff-seconds 1
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/verify_staged_schemas.py" \
    --sources "$AGENT1_V3_SOURCE_CONFIG" --lock "$lock" --destination "$destination" --output "$schemas"
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" "$PHASE04_DIR/scripts/finalize_acquisition.py" \
    --sources "$AGENT1_V3_SOURCE_CONFIG" --lock "$lock" --download-manifest "$download" \
    --schema-audit "$schemas" --destination "$destination" --code-commit "$AGENT1_V3_EXPECTED_COMMIT" \
    --output "$receipt"
ln -s "attempts/$AGENT1_V3_ATTEMPT_ID/hf.receipt.json" "$AGENT1_V3_PHASE0_DIR/hf_acquisition_receipt.json"
unset HF_TOKEN
echo "AGENT1_V3_HF_ACQUISITION_RECEIPT=$receipt"
