#!/usr/bin/env bash
# Close a checksum-bound sentinel handoff only after the serial queue is empty.
set -eEuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT COORD_ROOT PIPELINE_ROOT" >&2
  exit 2
fi
run=$(realpath "$1")
coord=$(realpath "$2")
pipeline=$(realpath "$3")
helper="$pipeline/scripts/finalize_signature_sentinel_cutover.py"
request="$run/dedup_acceleration_takeover_request.json"
arm="$run/dedup_acceleration_takeover_arm.json"
stop="$run/dedup_acceleration_sentinel_stop.json"
queue="$run/dedup_acceleration_sentinel_queue_evidence.json"
cutover="$run/dedup_acceleration_cutover.json"
manifest="$run/release-pre-dedup/manifests/combined_manifest.json"
[[ -f "$helper" && -f "$request" && -f "$arm" && -f "$stop" && -f "$manifest" ]] || { echo "sentinel evidence is incomplete" >&2; exit 1; }
[[ ! -e "$queue" && ! -e "$cutover" ]] || { echo "sentinel finalization artifacts already exist" >&2; exit 1; }

mapfile -t active < <(squeue -h -u fffoivos -p debug -o '%j' | grep -E '^a1v5-signature-chain-r[0-9]+$' || true)
(( ${#active[@]} == 0 )) || { echo "legacy signature successor exists: ${active[*]}" >&2; exit 1; }
python3 - "$queue" <<'PY'
import json, os, sys, tempfile
path = sys.argv[1]
payload = {"schema_version":"agent1_v5_dedup_acceleration_sentinel_queue_evidence_v1", "status":"passed", "debug_signature_queue_empty":True, "legacy_successor_present":False}
fd, temporary = tempfile.mkstemp(prefix=".queue-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True); handle.write("\n")
os.link(temporary, path); os.unlink(temporary)
PY
python3 "$helper" --request "$request" --arm-receipt "$arm" \
  --stop-receipt "$stop" --queue-evidence "$queue" --combined-manifest "$manifest" --output "$cutover"
