#!/usr/bin/env bash
# Submit, receipt-bind, and release one held normal-partition signature array.
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
runner="$pipeline/slurm/agent1_v5_eiger/normal_signature_runner.sh"
benchmark="$run/dedup_acceleration_benchmark.json"
cutover="$run/dedup_acceleration_cutover.json"
audit="$run/dedup_full_input_audit.json"
chunks="$run/dedup_acceleration_chunks.json"
submission="$run/dedup_acceleration_submission.json"
authorization="$run/dedup_acceleration_release_authorization.json"
journal="$run/dedup_acceleration_submission.pending.json"
evidence="$run/dedup_acceleration_submission_job_evidence.json"

[[ -x "$runner" && -f "$helper" && -f "$benchmark" && -f "$cutover" && -f "$audit" && -f "$chunks" ]]
[[ ! -e "$submission" && ! -e "$authorization" && ! -e "$journal" && ! -e "$evidence" ]] || {
  echo "submission artifacts already exist; use receipt-bound recovery, not a second submission" >&2
  exit 2
}

workers="$(jq -er 'select(.schema_version == "agent1_v5_dedup_acceleration_benchmark_v1" and .status == "passed" and .approved == true) | .selected_workers | select(. == 4 or . == 5)' "$benchmark")"
benchmark_sha="$(sha256sum "$benchmark" | awk '{print $1}')"
cutover_sha="$(sha256sum "$cutover" | awk '{print $1}')"
audit_sha="$(sha256sum "$audit" | awk '{print $1}')"
chunks_sha="$(sha256sum "$chunks" | awk '{print $1}')"
runner_sha="$(sha256sum "$runner" | awk '{print $1}')"
last_chunk="$(jq -er --argjson workers "$workers" --arg benchmark "$benchmark_sha" --arg cutover "$cutover_sha" --arg audit "$audit_sha" --arg runner "$runner_sha" '
  select(.schema_version == "agent1_v5_dedup_acceleration_chunk_plan_v1" and .status == "passed"
    and .selected_workers == $workers and .benchmark_receipt_sha256 == $benchmark
    and .cutover_receipt_sha256 == $cutover and .full_input_audit_sha256 == $audit
    and .runner_sha256 == $runner) | .last_chunk | select(. >= 0 and floor == .)' "$chunks")"
array_spec="0-${last_chunk}%1"
nonce="${SUBMISSION_NONCE_OVERRIDE:-$(tr -d '-' < /proc/sys/kernel/random/uuid)}"
[[ "$nonce" =~ ^[A-Za-z0-9._-]{12,128}$ ]] || { echo "invalid submission nonce" >&2; exit 2; }

write_json_exclusive() {
  local output="$1"
  shift
  python3 - "$output" "$@" <<'PY'
import json, os, sys, tempfile
path, *values = sys.argv[1:]
if os.path.exists(path):
    raise SystemExit(f"refusing to overwrite {path}")
payload = json.loads(values[0])
fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.link(temporary, path)
os.unlink(temporary)
PY
}

verify_held_array() {
  local job_id="$1" raw
  raw="$(scontrol show job -o "$job_id")"
  [[ "$raw" == *"UserId=fffoivos("* && "$raw" == *"Account=a0140 "* && "$raw" == *"Partition=normal "* ]] || return 1
  [[ "$raw" == *"JobName=a1v5-signature-normal-c${workers} "* && "$raw" == *"Comment=agent1-v5-dedup-accel:${nonce}"* ]] || return 1
  [[ "$raw" == *"JobState=PENDING "* && "$raw" == *"Reason=JobHeldUser "* ]] || return 1
  [[ "$raw" == *"ArrayTaskId=${array_spec} "* && "$raw" == *"StdOut=${coord}/slurm/"* && "$raw" == *"StdErr=${coord}/slurm/"* ]] || return 1
  printf '%s\n' "$raw"
}

array_job_id=""
armed=0
cleanup() {
  local raw
  if (( armed != 1 || -z "$array_job_id" )); then
    return
  fi
  if raw="$(verify_held_array "$array_job_id" 2>/dev/null)"; then
    scancel "$array_job_id" || true
  else
    echo "refusing cleanup: held-array identity no longer matches $array_job_id" >&2
  fi
}
trap cleanup EXIT INT TERM ERR

pending_payload="$(jq -cn --arg run "$run" --arg coord "$coord" --arg pipeline "$pipeline" --arg nonce "$nonce" --arg benchmark "$benchmark_sha" --arg cutover "$cutover_sha" --arg audit "$audit_sha" --arg chunks "$chunks_sha" --arg runner "$runner_sha" --arg array "$array_spec" --argjson workers "$workers" '{schema_version:"agent1_v5_dedup_acceleration_submission_pending_v1",status:"pending",run_root:$run,coord_root:$coord,pipeline_root:$pipeline,submission_nonce:$nonce,benchmark_receipt_sha256:$benchmark,cutover_receipt_sha256:$cutover,full_input_audit_sha256:$audit,chunk_plan_sha256:$chunks,runner_sha256:$runner,array_spec:$array,selected_workers:$workers}')"
write_json_exclusive "$journal" "$pending_payload"

submitted="$(
  sbatch --parsable --hold --uenv-passthrough=ignore \
    --account=a0140 --partition=normal \
    --comment="agent1-v5-dedup-accel:${nonce}" \
    --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G \
    --time=11:30:00 --signal=B:USR1@900 --array="$array_spec" \
    --job-name="a1v5-signature-normal-c${workers}" \
    --output="$coord/slurm/%x-%A_%a.out" --error="$coord/slurm/%x-%A_%a.err" \
    --export=ALL,RUN_ROOT="$run",PIPELINE_ROOT="$pipeline",CHUNK_PLAN="$chunks",CHUNK_PLAN_SHA256="$chunks_sha",FULL_INPUT_AUDIT="$audit",WORKERS="$workers",SUBMISSION_NONCE="$nonce",SUBMISSION_RECEIPT="$submission",RELEASE_AUTHORIZATION="$authorization",STOP_SENTINEL="$run/dedup_acceleration.stop" \
    "$runner"
)"
array_job_id="${submitted%%;*}"
[[ "$array_job_id" =~ ^[0-9]+$ ]] || { echo "unparseable sbatch ID: $submitted" >&2; exit 2; }
armed=1
raw="$(verify_held_array "$array_job_id")" || { echo "held-array identity validation failed" >&2; exit 1; }

evidence_payload="$(jq -cn --arg id "$array_job_id" --arg nonce "$nonce" --arg array "$array_spec" --arg coord "$(realpath "$coord")" --arg raw "$raw" --argjson workers "$workers" '{schema_version:"agent1_v5_dedup_acceleration_held_array_evidence_v1",status:"passed",array_job_id:$id,submission_nonce:$nonce,owner:"fffoivos",account:"a0140",partition:"normal",job_name:("a1v5-signature-normal-c" + ($workers|tostring)),array_spec:$array,coord_root:$coord,state:"PENDING",reason:"JobHeldUser",scontrol_raw:$raw}' )"
write_json_exclusive "$evidence" "$evidence_payload"

"$python" "$helper" record-submission \
  --run-root "$run" --benchmark-receipt "$benchmark" --cutover-receipt "$cutover" \
  --full-input-audit "$audit" --chunk-plan "$chunks" --runner "$runner" \
  --job-evidence "$evidence" --array-job-id "$array_job_id" --submission-nonce "$nonce" \
  --array-spec "$array_spec" --coord-root "$coord" --output "$submission"
"$python" "$helper" authorize-release --submission-receipt "$submission" \
  --array-job-id "$array_job_id" --submission-nonce "$nonce" --output "$authorization"

scontrol release "$array_job_id"
armed=0
release_payload="$(jq -cn --arg id "$array_job_id" --arg nonce "$nonce" '{schema_version:"agent1_v5_dedup_acceleration_release_observation_v1",status:"passed",array_job_id:$id,submission_nonce:$nonce,release_requested:true}')"
write_json_exclusive "$run/dedup_acceleration_release_observation.json" "$release_payload"
printf 'submitted and released accelerated array %s\n' "$array_job_id"
