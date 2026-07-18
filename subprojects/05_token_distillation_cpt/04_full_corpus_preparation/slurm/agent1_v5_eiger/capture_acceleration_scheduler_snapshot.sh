#!/usr/bin/env bash
# Capture the live scheduler facts used by the acceleration preflight.
set -eEuo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_ROOT OUTPUT_JSON" >&2
  exit 2
fi
run="$1"
output="$2"
[[ ! -e "$output" ]] || { echo "refusing to overwrite scheduler snapshot" >&2; exit 2; }

partition="$(scontrol show partition debug -o)"
[[ "$partition" == *"PartitionName=debug "* && "$partition" == *"QoS=debug-qos "* ]] || {
  echo "debug partition does not expose debug-qos" >&2; exit 1;
}
qos="$(sacctmgr -n -P show qos debug-qos format=Name,MaxJobsPU,MaxSubmitJobsPU | head -n 1)"
IFS='|' read -r qos_name max_jobs max_submit <<<"$qos"
[[ "$qos_name" == debug-qos && "$max_jobs" == 1 && "$max_submit" == 2 ]] || {
  echo "debug-qos limits differ from fence requirements: $qos" >&2; exit 1;
}
mapfile -t jobs < <(squeue -h -u fffoivos -p debug -o '%i|%j|%T|%a')
(( ${#jobs[@]} == 1 )) || { echo "expected exactly one debug job for fffoivos" >&2; exit 1; }
IFS='|' read -r job_id job_name job_state job_account <<<"${jobs[0]}"
[[ "$job_name" =~ ^a1v5-signature-chain-r[0-9]+$ && "$job_state" == RUNNING && "$job_account" == a0140 ]] || {
  echo "unexpected active debug job: ${jobs[0]}" >&2; exit 1;
}
raw="$(scontrol show job -o "$job_id")"
effective_qos="$(sed -n 's/.* QOS=\([^ ]*\).*/\1/p' <<<"$raw")"
python3 - "$output" "$run" "$job_id" "$job_name" "$effective_qos" "$partition" "$qos" <<'PY'
import json, os, sys, tempfile
path, run, job_id, job_name, effective_qos, partition, qos = sys.argv[1:]
payload = {
    "schema_version": "agent1_v5_dedup_acceleration_scheduler_snapshot_v1",
    "status": "passed",
    "run_root": os.path.realpath(run),
    "partition": "debug",
    "qos": "debug-qos",
    "account": "a0140",
    "user": "fffoivos",
    "max_jobs_per_user": 1,
    "max_submit_jobs_per_user": 2,
    "effective_legacy_qos": effective_qos,
    "legacy_job_id": job_id,
    "legacy_job_name": job_name,
    "partition_raw": partition,
    "qos_raw": qos,
}
fd, temporary = tempfile.mkstemp(prefix=".scheduler-", dir=os.path.dirname(path))
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.link(temporary, path)
os.unlink(temporary)
PY
