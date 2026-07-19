#!/usr/bin/env bash
# Capture immutable task-level sacct evidence for one exact signature array.
set -eEuo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 ARRAY_JOB_ID ARRAY_SPEC ATTEMPT_ID EXPECTED_STATE EXPECTED_EXIT OUTPUT" >&2
  exit 2
fi

array_job_id="$1"
array_spec="$2"
attempt_id="$3"
expected_state="$4"
expected_exit="$5"
output="$6"

[[ "$array_job_id" =~ ^[0-9]+$ ]]
[[ "$array_spec" =~ ^[0-9]+-[0-9]+%[0-9]+$ ]]
[[ "$attempt_id" =~ ^[a-z0-9][a-z0-9._-]{5,63}$ ]]
case "$expected_state:$expected_exit" in
  COMPLETED:0:0|FAILED:127:0) ;;
  *) echo "unsupported expected terminal state: $expected_state/$expected_exit" >&2; exit 2 ;;
esac
[[ ! -e "$output" ]] || { echo "immutable evidence already exists: $output" >&2; exit 2; }

temporary="$(mktemp "${output}.sacct.XXXXXX")"
trap 'rm -f "$temporary"' EXIT
sacct -j "$array_job_id" --noheader --parsable2 \
  --format=JobID,State,ExitCode,Elapsed,JobName,Account,Partition > "$temporary"

python3 - "$temporary" "$output" "$array_job_id" "$array_spec" "$attempt_id" "$expected_state" "$expected_exit" <<'PY'
import datetime
import json
import os
import re
import sys
import tempfile

source, output, array_job_id, array_spec, attempt_id, expected_state, expected_exit = sys.argv[1:]
raw_range = array_spec.split("%", 1)[0]
first, last = [int(value) for value in raw_range.split("-", 1)]
expected = list(range(first, last + 1))
pattern = re.compile(r"^%s_([0-9]+)$" % re.escape(array_job_id))
tasks = []
with open(source, encoding="utf-8") as handle:
    for raw in handle:
        fields = raw.rstrip("\n").split("|")
        if len(fields) < 7:
            continue
        match = pattern.fullmatch(fields[0])
        if not match:
            continue
        tasks.append({
            "job_id": fields[0],
            "task_index": int(match.group(1)),
            "state": fields[1].split()[0],
            "exit_code": fields[2],
            "elapsed": fields[3],
            "job_name": fields[4],
            "account": fields[5],
            "partition": fields[6],
        })
tasks.sort(key=lambda task: task["task_index"])
if [task["task_index"] for task in tasks] != expected:
    raise SystemExit("scheduler evidence does not close expected array tasks")
for task in tasks:
    if task["state"] != expected_state or task["exit_code"] != expected_exit:
        raise SystemExit("scheduler task has unexpected terminal state: %r" % task)
    if task["account"] != "a0140" or task["partition"] != "normal":
        raise SystemExit("scheduler task identity drift: %r" % task)
payload = {
    "schema_version": "agent1_v5_dedup_acceleration_array_execution_evidence_v1",
    "status": "passed",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "array_job_id": array_job_id,
    "array_spec": array_spec,
    "attempt_id": attempt_id,
    "expected_state": expected_state,
    "expected_exit_code": expected_exit,
    "tasks": tasks,
}
directory = os.path.dirname(output)
fd, pending = tempfile.mkstemp(prefix=".array-evidence-", dir=directory)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, sort_keys=True)
    handle.write("\n")
os.link(pending, output)
os.unlink(pending)
PY

