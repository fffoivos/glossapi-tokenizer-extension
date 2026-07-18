#!/usr/bin/env bash
# Fence the legacy chain only when its *effective* QOS provides the documented
# two-submission boundary.  A mismatch is a hard stop: this script never
# guesses that a held job will block a successor.
set -eEuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RUN_ROOT COORD_ROOT PIPELINE_ROOT" >&2
  exit 2
fi
run="$1"
coord="$2"
pipeline="$3"
python="$coord/runtime/venv/bin/python"
helper="$pipeline/scripts/agent1_v5_dedup_acceleration.py"
preflight="$run/dedup_acceleration_preflight.json"
manifest="$run/release-pre-dedup/manifests/combined_manifest.json"
boundary="$run/dedup_acceleration_boundary_staging.json"
fence_evidence="$run/dedup_acceleration_fence_evidence.json"
successor_evidence="$run/dedup_acceleration_successor_evidence.json"
cutover="$run/dedup_acceleration_cutover.json"
[[ -f "$helper" && -f "$preflight" && -f "$manifest" ]]
[[ ! -e "$boundary" && ! -e "$fence_evidence" && ! -e "$successor_evidence" && ! -e "$cutover" ]] || {
  echo "cutover artifacts already exist; use receipt-bound recovery" >&2
  exit 2
}

jq -e --arg run "$(realpath "$run")" --arg pipeline "$(realpath "$pipeline")" '
  .schema_version == "agent1_v5_dedup_acceleration_preflight_v1" and .status == "passed"
  and .explicit_fence_approval == true and .blocking_acceleration_findings == []
  and .run_root == $run and .pipeline_root == $pipeline' "$preflight" >/dev/null

snapshot_policy() {
  local partition qos
  partition="$(scontrol show partition debug -o)"
  [[ "$partition" == *"PartitionName=debug "* && "$partition" == *"QoS=debug-qos "* ]] || return 1
  qos="$(sacctmgr -n -P show qos debug-qos format=Name,MaxJobsPU,MaxSubmitJobsPU | head -n 1)"
  [[ "$qos" == "debug-qos|1|2" ]]
}

chain_job=""
chain_rank=""
verify_chain_job() {
  local job_id="$1" rank="$2" raw
  raw="$(scontrol show job -o "$job_id")"
  [[ "$raw" == *"JobId=$job_id "* && "$raw" == *"JobName=a1v5-signature-chain-r${rank} "* ]] || return 1
  [[ "$raw" == *"UserId=fffoivos("* && "$raw" == *"Account=a0140 "* && "$raw" == *"Partition=debug "* ]] || return 1
  # This is the crucial protection.  The partition default is insufficient:
  # a chain job admitted under normal QOS has no MaxSubmitJobsPU fence.
  [[ "$raw" == *"QOS=debug-qos "* && "$raw" == *"JobState=RUNNING "* ]] || return 1
  [[ "$raw" == *"StdOut=${coord}/slurm/"* && "$raw" == *"StdErr=${coord}/slurm/"* ]] || return 1
  printf '%s\n' "$raw"
}

mapfile -t active < <(squeue -h -u fffoivos -p debug -o '%i|%j|%T|%a')
(( ${#active[@]} == 1 )) || { echo "expected exactly one submitted debug job" >&2; exit 1; }
IFS='|' read -r observed_id observed_name observed_state observed_account <<<"${active[0]}"
[[ "$observed_name" =~ ^a1v5-signature-chain-r([0-9]+)$ ]] || { echo "unexpected debug job: $observed_name" >&2; exit 1; }
chain_rank="${BASH_REMATCH[1]}"
(( chain_rank < 430 )) || { echo "no signature rank remains to fence" >&2; exit 1; }
[[ "$observed_state" == RUNNING && "$observed_account" == a0140 ]] || { echo "chain job is not running on a0140" >&2; exit 1; }
snapshot_policy || { echo "debug partition/QOS fence policy no longer matches preflight" >&2; exit 1; }
verify_chain_job "$observed_id" "$chain_rank" >/dev/null || {
  echo "legacy job lacks effective debug-qos fence eligibility; no cutover was attempted" >&2
  exit 1
}
chain_job="$observed_id"

nonce="$(tr -d '-' < /proc/sys/kernel/random/uuid)"
fence_job=""
armed=0
verify_held_fence() {
  local job_id="$1" raw
  raw="$(scontrol show job -o "$job_id")"
  [[ "$raw" == *"UserId=fffoivos("* && "$raw" == *"Account=a0140 "* && "$raw" == *"Partition=debug "* ]] || return 1
  [[ "$raw" == *"QOS=debug-qos "* && "$raw" == *"JobName=a1v5-signature-chain-fence "* ]] || return 1
  [[ "$raw" == *"Comment=agent1-v5-dedup-fence:${nonce}"* && "$raw" == *"JobState=PENDING "* && "$raw" == *"Reason=JobHeldUser "* ]] || return 1
  [[ "$raw" == *"StdOut=${coord}/slurm/"* && "$raw" == *"StdErr=${coord}/slurm/"* ]] || return 1
  printf '%s\n' "$raw"
}
cleanup() {
  if (( armed == 1 )) && [[ -n "$fence_job" ]] && verify_held_fence "$fence_job" >/dev/null 2>&1; then
    scancel "$fence_job" || true
  fi
}
trap cleanup EXIT INT TERM ERR

submitted="$(sbatch --parsable --hold --uenv-passthrough=ignore --account=a0140 --partition=debug --qos=debug-qos \
  --comment="agent1-v5-dedup-fence:${nonce}" --nodes=1 --ntasks=1 --cpus-per-task=1 --time=00:05:00 \
  --job-name=a1v5-signature-chain-fence --output="$coord/slurm/%x-%j.out" --error="$coord/slurm/%x-%j.err" --wrap 'true')"
fence_job="${submitted%%;*}"
[[ "$fence_job" =~ ^[0-9]+$ ]] || { echo "unparseable held fence job ID" >&2; exit 1; }
armed=1
verify_held_fence "$fence_job" >/dev/null || { echo "held fence identity drift" >&2; exit 1; }

# The successor helper is expected to exit nonzero only because the held fence
# consumes the final permitted debug-qos submission slot.  We never cancel it.
while squeue -h -j "$chain_job" | grep -q .; do
  verify_held_fence "$fence_job" >/dev/null || { echo "held fence disappeared" >&2; exit 1; }
  sleep 20
done
receipt="$run/60-dedup/minhash-signatures/receipts/$(printf '%06d' "$chain_rank").json"
err="$coord/slurm/a1v5-signature-chain-r${chain_rank}-${chain_job}.err"
[[ -f "$receipt" && -f "$err" ]] || { echo "legacy completion has no receipt or stderr evidence" >&2; exit 1; }
grep -Eqi 'submit.*fail|job violates.*(qos|limit)|association.*limit' "$err" || {
  echo "legacy exit was not the expected submission-limit rejection" >&2; exit 1;
}
next=$((chain_rank + 1))
! squeue -h -u fffoivos -p debug -o '%j' | grep -qx "a1v5-signature-chain-r${next}" || {
  echo "unexpected legacy successor exists" >&2; exit 1;
}
"$python" "$helper" validate-boundary --run-root "$run" --combined-manifest "$manifest" \
  --through-rank "$chain_rank" --fence-job-id "$fence_job" --final-legacy-job-id "$chain_job" --output "$boundary"

raw="$(verify_held_fence "$fence_job")"
scancel "$fence_job"
until ! squeue -h -j "$fence_job" | grep -q .; do sleep 2; done
! squeue -h -u fffoivos -p debug -o '%j' | grep -Eq '^a1v5-signature-chain-(r[0-9]+|fence)$' || {
  echo "debug signature queue did not empty after fence cleanup" >&2; exit 1;
}
python3 - "$fence_evidence" "$fence_job" "$chain_job" "$raw" <<'PY'
import json, os, sys, tempfile
path, fence, legacy, raw = sys.argv[1:]
payload = {"schema_version":"agent1_v5_dedup_acceleration_fence_evidence_v1","status":"passed","fence_job_id":fence,"final_legacy_job_id":legacy,"fence_cleanup_result":"cancelled","debug_signature_queue_empty":True,"scontrol_raw":raw}
fd, temp = tempfile.mkstemp(prefix=".fence-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as out: json.dump(payload, out, sort_keys=True); out.write("\n")
os.link(temp, path); os.unlink(temp)
PY
python3 - "$successor_evidence" "$err" <<'PY'
import hashlib, json, os, sys, tempfile
path, stderr = sys.argv[1:]
payload = {"schema_version":"agent1_v5_dedup_acceleration_successor_rejection_v1","status":"passed","expected_successor_rejection":True,"no_successor":True,"stderr_sha256":hashlib.sha256(open(stderr,"rb").read()).hexdigest()}
fd, temp = tempfile.mkstemp(prefix=".successor-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as out: json.dump(payload, out, sort_keys=True); out.write("\n")
os.link(temp, path); os.unlink(temp)
PY
"$python" "$helper" finalize-cutover --preflight "$preflight" --boundary "$boundary" \
  --fence-evidence "$fence_evidence" --successor-evidence "$successor_evidence" --output "$cutover"
armed=0
printf 'cutover receipt finalized at first missing rank %s\n' "$next"
