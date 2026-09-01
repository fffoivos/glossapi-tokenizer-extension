#!/usr/bin/env bash
#SBATCH --job-name=full8b-prequeued-holder
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_PREQUEUED_MANIFEST:?set prequeued manifest}"
: "${FULL8_PREQUEUE_PERMIT:?set launch-permit path}"
: "${FULL8_PREQUEUE_SOURCE_SEGMENT:?set source segment}"
: "${FULL8_PREQUEUE_SOURCE_TRAIN_JOB:?set source train job}"
: "${FULL8_SEGMENT_ID:?set target segment}"
: "${FULL8_START_ITERATION:?set exact segment start}"
: "${FULL8_END_ITERATION:?set exact segment end}"
: "${FULL8_MIN_TRAIN_SECONDS:?set conservative required wall time}"
: "${FULL8_MAX_HOLD_SECONDS:=5400}"
: "${FULL8_ALLOCATION_RESERVE_SECONDS:?set frozen allocation reserve}"
[[ "${SLURM_JOB_PARTITION:-}" == normal ]] || {
  echo "prequeued holder must run on normal" >&2
  exit 2
}
holder_started_epoch=$(date +%s)

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

holder_root="$FULL8_RUN_ROOT/orchestration/prequeued_holders/job_${SLURM_JOB_ID}"
mkdir -p "$holder_root"

manifest_bound=0
for _ in $(seq 1 60); do
if /usr/bin/python3.11 - "$FULL8_PREQUEUED_MANIFEST" "$SLURM_JOB_ID" \
  "$FULL8_PREQUEUE_SOURCE_SEGMENT" "$FULL8_PREQUEUE_SOURCE_TRAIN_JOB" \
  "$FULL8_SEGMENT_ID" "$FULL8_START_ITERATION" "$FULL8_END_ITERATION" \
  "$FULL8_PREQUEUE_PERMIT" "$FULL8_ALLOCATION_RESERVE_SECONDS" <<'PY'
import json, os, sys
manifest, job, source_segment, source_job, segment, start, end, permit, reserve = sys.argv[1:]
value = json.load(open(manifest, encoding="utf-8"))
if value.get("schema_version") != "apertus_full_8b_prequeued_launch_graph_v1":
    raise SystemExit("prequeued manifest schema drift")
if value.get("status") != "submitted":
    raise SystemExit("prequeued manifest is not submitted")
matches = [row for row in value.get("segments", []) if int(row["segment_id"]) == int(segment)]
if len(matches) != 1:
    raise SystemExit("prequeued manifest must contain exactly one target segment")
row = matches[0]
expected = {
    "train_job": job,
    "source_segment_id": int(source_segment),
    "source_train_job": source_job,
    "start_iteration": int(start),
    "end_iteration": int(end),
    "allocation_reserve_seconds": int(reserve),
    "permit_path": os.path.realpath(permit),
}
for key, expected_value in expected.items():
    actual = row.get(key)
    if key == "permit_path" and actual is not None:
        actual = os.path.realpath(actual)
    if actual != expected_value:
        raise SystemExit(f"prequeued manifest binding drift for {key}: {actual!r} != {expected_value!r}")
PY
then
  manifest_bound=1
  break
fi
sleep 2
done
(( manifest_bound == 1 )) || {
  echo "prequeued manifest did not bind this allocation within 120s" >&2
  exit 2
}

while [[ ! -s "$FULL8_PREQUEUE_PERMIT" ]]; do
  elapsed=$(( $(date +%s) - holder_started_epoch ))
  if (( elapsed >= FULL8_MAX_HOLD_SECONDS )); then
    echo "prequeued holder timed out waiting for permit after ${elapsed}s" >&2
    exit 3
  fi
  sleep 15
done

/usr/bin/python3.11 - "$FULL8_PREQUEUE_PERMIT" "$SLURM_JOB_ID" \
  "$FULL8_PREQUEUE_SOURCE_SEGMENT" "$FULL8_PREQUEUE_SOURCE_TRAIN_JOB" \
  "$FULL8_SEGMENT_ID" "$FULL8_START_ITERATION" "$FULL8_END_ITERATION" <<'PY'
import json, sys
permit, job, source_segment, source_job, segment, start, end = sys.argv[1:]
value = json.load(open(permit, encoding="utf-8"))
expected = {
    "schema_version": "apertus_full_8b_prequeued_train_permit_v1",
    "status": "passed",
    "prequeued_train_job": job,
    "source_segment_id": int(source_segment),
    "source_train_job": source_job,
    "target_segment_id": int(segment),
    "start_iteration": int(start),
    "end_iteration": int(end),
}
for key, expected_value in expected.items():
    if value.get(key) != expected_value:
        raise SystemExit(f"launch permit binding drift for {key}")
checkpoint_receipt = value.get("source_checkpoint_receipt")
if not checkpoint_receipt:
    raise SystemExit("launch permit lacks source checkpoint receipt")
PY

remaining=$(squeue -h -j "$SLURM_JOB_ID" -o '%L')
remaining_seconds=$(/usr/bin/python3.11 - "$remaining" <<'PY'
import sys
value=sys.argv[1].strip()
days=0
if "-" in value:
    day,value=value.split("-",1); days=int(day)
parts=[int(part) for part in value.split(":")]
if len(parts)==3: hours,minutes,seconds=parts
elif len(parts)==2: hours=0; minutes,seconds=parts
else: raise SystemExit("invalid Slurm remaining-time format")
print(days*86400+hours*3600+minutes*60+seconds)
PY
)
required_remaining=$(( FULL8_MIN_TRAIN_SECONDS + FULL8_ALLOCATION_RESERVE_SECONDS ))
if (( remaining_seconds < required_remaining )); then
  echo "insufficient allocation time after hold: ${remaining_seconds}s < ${required_remaining}s (training plus reserve)" >&2
  exit 4
fi

exec bash "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"
