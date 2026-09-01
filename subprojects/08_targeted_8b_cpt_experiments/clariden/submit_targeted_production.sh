#!/usr/bin/env bash
set -euo pipefail

: "${TARGET8_EXPERIMENT:?set A or B}"
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_STAGE_ROOT:?set frozen experiment stage root}"
: "${FULL8_RUN_ROOT:?set new production run root}"
: "${FULL8_INITIAL_MEGATRON:?set verified initialization/checkpoint root}"
: "${FULL8_PRELAUNCH_ROOT:?set completed prelaunch root}"
: "${FULL8_SELECTED_PROFILE:?set selected profile receipt}"
: "${FULL8_LAUNCH_GATE:?set targeted launch gate}"
: "${FULL8_OPERATIONAL_LAUNCH_GATE:?set operational launch gate}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set pinned Clariden leaf switch}"

DRY_RUN=${DRY_RUN:-1}
[[ "$TARGET8_EXPERIMENT" == A || "$TARGET8_EXPERIMENT" == B ]] || { echo "TARGET8_EXPERIMENT must be A or B" >&2; exit 2; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }

if [[ "$TARGET8_EXPERIMENT" == B ]]; then
  /usr/bin/python3.11 - "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/experiment_b_recipe.json" <<'PY'
import json
import sys

recipe = json.load(open(sys.argv[1]))
if recipe.get("launch_authorized") is not True:
    raise SystemExit(
        "Experiment B was retired by the owner; its preserved builder and "
        "receipts are not launch authorization"
    )
PY
fi

RECIPE=${FULL8_RECIPE:-$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json}
PROFILES=${FULL8_PROFILES:-$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json}
PREQUEUE_SCHEDULE=${FULL8_PREQUEUE_SCHEDULE:-$FULL8_STAGE_ROOT/contracts/prequeue_schedule.json}
MANIFEST="$FULL8_RUN_ROOT/submissions/targeted_launch_graph.json"
PREQUEUED_MANIFEST="$FULL8_RUN_ROOT/orchestration/prequeued_launch_graph.json"

/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

readarray -t selected < <(/usr/bin/python3.11 - "$TARGET8_EXPERIMENT" "$FULL8_SELECTED_PROFILE" "$FULL8_LAUNCH_GATE" "$RECIPE" <<'PY'
import json, sys
from pathlib import Path

experiment = sys.argv[1]
profile = json.load(open(sys.argv[2]))
gate = json.load(open(sys.argv[3]))
if profile.get("schema_version") != "apertus_full_8b_selected_execution_profile_v1" or profile.get("status") != "frozen":
    raise SystemExit("selected profile is not frozen")
if gate.get("schema_version") != "apertus_full_8b_launch_gate_v1" or gate.get("status") != "passed":
    raise SystemExit("launch gate is not passed")
if gate.get("experiment") != experiment or profile["selection"] != gate["selected_profile"]:
    raise SystemExit("launch gate experiment/profile drift")
if Path(profile["recipe"]["path"]).resolve() != Path(sys.argv[4]).resolve():
    raise SystemExit("selected recipe binding drift")
selection = profile["selection"]
boundaries = list(map(int, selection["segment_boundaries"]))
if selection["profile_id"] != "dp32_16node" or selection["nodes"] != 16 or selection["data_parallel"] != 32:
    raise SystemExit("production geometry drift")
if experiment == "A":
    if len(boundaries) != 3 or boundaries[0] != 0:
        raise SystemExit("A segment geometry drift")
    segment = 0
else:
    if boundaries[:2] != [0, 9536] or len(boundaries) != 3:
        raise SystemExit("B continuation geometry drift")
    segment = 1
print(selection["profile_id"])
print(selection["nodes"])
print(segment)
print(boundaries[segment])
print(boundaries[segment + 1])
PY
)
/usr/bin/python3.11 - "$FULL8_OPERATIONAL_LAUNCH_GATE" "$FULL8_CODE_ROOT" "$FULL8_OPS_ROOT" <<'PY'
import json, os, sys

value = json.load(open(sys.argv[1]))
if value.get("schema_version") != "apertus_full_8b_operational_launch_gate_v1" or value.get("status") != "passed":
    raise SystemExit("operational launch gate is not passed")
if os.path.realpath(value.get("scientific_root", "")) != os.path.realpath(sys.argv[2]):
    raise SystemExit("operational/scientific root drift")
if os.path.realpath(value.get("operational_root", "")) != os.path.realpath(sys.argv[3]):
    raise SystemExit("operational root drift")
PY

profile_id=${selected[0]}
nodes=${selected[1]}
segment=${selected[2]}
start=${selected[3]}
end=${selected[4]}
train_exclude=$("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$FULL8_TRAIN_LEAF_SWITCH" "$nodes")
[[ -n "$train_exclude" ]] || { echo "empty leaf exclusion" >&2; exit 2; }

prequeue_exports=""
launch_initial_megatron="$FULL8_INITIAL_MEGATRON"
if [[ "$TARGET8_EXPERIMENT" == A ]]; then
  [[ -s "$PREQUEUE_SCHEDULE" ]] || { echo "A prequeue schedule missing" >&2; exit 2; }
  prequeue_exports=",FULL8_PREQUEUED_MANIFEST=$PREQUEUED_MANIFEST,FULL8_PREQUEUE_SCHEDULE=$PREQUEUE_SCHEDULE"
else
  launch_initial_megatron="$FULL8_RUN_ROOT/checkpoints"
fi
common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_OPS_ROOT=$FULL8_OPS_ROOT,FULL8_OPS_BUNDLE_RECEIPT=$FULL8_OPS_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_RUN_ROOT=$FULL8_RUN_ROOT,FULL8_INITIAL_MEGATRON=$launch_initial_megatron,FULL8_PRELAUNCH_ROOT=$FULL8_PRELAUNCH_ROOT,FULL8_SELECTED_PROFILE=$FULL8_SELECTED_PROFILE,FULL8_EXECUTION_PROFILE=$profile_id,FULL8_LAUNCH_GATE=$FULL8_LAUNCH_GATE,FULL8_RECIPE=$RECIPE,FULL8_PROFILES=$PROFILES,FULL8_TRAIN_LEAF_SWITCH=$FULL8_TRAIN_LEAF_SWITCH$prequeue_exports"
job_tag=$(printf '%s' "$TARGET8_EXPERIMENT" | tr '[:upper:]' '[:lower:]')
train_command=(sbatch --uenv-passthrough=ignore --parsable --partition=normal --time=12:00:00 --switches=1 \
  --exclude="$train_exclude" --nodes="$nodes" --job-name="target8${job_tag}_s${segment}a0" \
  --output="$FULL8_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_SEGMENT_ID=$segment,FULL8_ATTEMPT=0,FULL8_START_ITERATION=$start,FULL8_END_ITERATION=$end,FULL8_LOAD_CHECKPOINT=$launch_initial_megatron,FULL8_RECOVERY_MODE=0" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")

test_result=$(sbatch --test-only "${train_command[@]:1}" 2>&1) || { echo "$test_result" >&2; exit 2; }
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'TRAIN_TEST_ONLY_OK %s\n' "$test_result"
  { printf 'TRAIN'; printf ' %q' "${train_command[@]}"; printf '\n'; }
  exit 0
fi

[[ "${CONFIRM_GPU_LAUNCH:-}" == "TARGETED8_CPT_${TARGET8_EXPERIMENT}" ]] || { echo "set CONFIRM_GPU_LAUNCH=TARGETED8_CPT_${TARGET8_EXPERIMENT}" >&2; exit 2; }
[[ ! -e "$FULL8_RUN_ROOT" ]] || { echo "run root already exists" >&2; exit 2; }
mkdir -p "$FULL8_RUN_ROOT/logs" "$FULL8_RUN_ROOT/submissions" "$FULL8_RUN_ROOT/orchestration/events" \
  "$FULL8_RUN_ROOT/orchestration/allocation_receipts"
if [[ "$TARGET8_EXPERIMENT" == B ]]; then
  : "${TARGET8_PARENT_CHECKPOINT_ROOT:?B requires parent checkpoint root}"
  source_checkpoint="$TARGET8_PARENT_CHECKPOINT_ROOT/iter_0009536"
  [[ -f "$source_checkpoint/.metadata" ]] || { echo "B parent checkpoint is incomplete" >&2; exit 2; }
  mkdir -p "$FULL8_RUN_ROOT/checkpoints"
  ln -s "$source_checkpoint" "$FULL8_RUN_ROOT/checkpoints/iter_0009536"
  printf '9536\n' >"$FULL8_RUN_ROOT/checkpoints/latest_checkpointed_iteration.txt"
fi

train=""
supervisor=""
cleanup_failed_launch() {
  set +e
  [[ -n "$supervisor" ]] && scancel "$supervisor"
  [[ -n "$train" ]] && scancel "$train"
  set -e
}
trap cleanup_failed_launch ERR
train=$("${train_command[@]}")

if [[ "$TARGET8_EXPERIMENT" == A ]]; then
  /usr/bin/python3.11 - "$PREQUEUED_MANIFEST" "$train" "$segment" "$start" "$end" <<'PY'
import datetime, json, os, sys, tempfile

out, job, segment, start, end = sys.argv[1:]
value = {
    "schema_version": "apertus_full_8b_prequeued_launch_graph_v1",
    "status": "submitted",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "segments": [{"segment_id": int(segment), "train_job": job, "start_iteration": int(start), "end_iteration": int(end), "role": "source"}],
}
os.makedirs(os.path.dirname(out), exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=".prequeue.", suffix=".partial", dir=os.path.dirname(out))
with os.fdopen(fd, "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, out)
PY
  graph_before=$(sha256sum "$PREQUEUED_MANIFEST" | awk '{print $1}')
  prequeue_common=(--scientific-root "$FULL8_CODE_ROOT" --scientific-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
    --ops-root "$FULL8_OPS_ROOT" --ops-receipt "$FULL8_OPS_BUNDLE_RECEIPT" --stage-root "$FULL8_STAGE_ROOT" \
    --run-root "$FULL8_RUN_ROOT" --initial-megatron "$FULL8_INITIAL_MEGATRON" --selected-profile "$FULL8_SELECTED_PROFILE" \
    --launch-gate "$FULL8_LAUNCH_GATE" --prelaunch-root "$FULL8_PRELAUNCH_ROOT" --recipe "$RECIPE" --profiles "$PROFILES" \
    --train-leaf-switch "$FULL8_TRAIN_LEAF_SWITCH" --manifest "$PREQUEUED_MANIFEST" --schedule "$PREQUEUE_SCHEDULE" \
    --source-segment 0 --source-train-job "$train" --target-segment 1 --minimum-train-seconds 36000 \
    --maximum-hold-seconds 6000 --eligible-after-minutes 500)
  /usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/prequeue_next_segment.py" "${prequeue_common[@]}" --test-only \
    --output "$FULL8_RUN_ROOT/orchestration/prequeue_test_only.json"
  [[ "$(sha256sum "$PREQUEUED_MANIFEST" | awk '{print $1}')" == "$graph_before" ]] || { echo "test-only mutated prequeue graph" >&2; exit 2; }
  /usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/prequeue_next_segment.py" "${prequeue_common[@]}" \
    --output "$FULL8_RUN_ROOT/orchestration/prequeue_submission.json"
fi

supervisor=$(sbatch --uenv-passthrough=ignore --parsable --partition=debug --time=00:20:00 --nodes=1 \
  --dependency="afterany:$train" --job-name="target8${job_tag}_supervise_s${segment}a0" \
  --output="$FULL8_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_RUN_ROOT/logs/%x-%j.err" --export="$common" \
  "$FULL8_OPS_ROOT/clariden/supervise_campaign_resource_aware.sbatch" "$segment" 0 "$start" "$train")

/usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/audit_submitted_job_resources.py" \
  --scientific-root "$FULL8_CODE_ROOT" --scientific-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --ops-root "$FULL8_OPS_ROOT" --ops-receipt "$FULL8_OPS_BUNDLE_RECEIPT" \
  --job "train=$train" --job "supervisor=$supervisor" \
  --output "$FULL8_RUN_ROOT/orchestration/allocation_receipts/initial_${train}_${supervisor}.json"

/usr/bin/python3.11 - "$MANIFEST" "$TARGET8_EXPERIMENT" "$train" "$supervisor" "$segment" "$start" "$end" \
  "$test_result" "$FULL8_CODE_ROOT" "$FULL8_CODE_BUNDLE_RECEIPT" "$FULL8_OPS_ROOT" "$FULL8_OPS_BUNDLE_RECEIPT" <<'PY'
import datetime, hashlib, json, os, sys, tempfile

out, experiment, train, supervisor, segment, start, end, test_result, code_root, code_receipt, ops_root, ops_receipt = sys.argv[1:]
def bundle(root, receipt):
    raw = open(receipt, "rb").read()
    value = json.loads(raw)
    return {"root": os.path.realpath(root), "receipt": os.path.realpath(receipt), "receipt_sha256": hashlib.sha256(raw).hexdigest(), "tree_sha256": value["tree_sha256"]}
value = {
    "schema_version": "apertus_targeted_8b_launch_graph_v1",
    "status": "submitted",
    "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "experiment": experiment,
    "scientific_bundle": bundle(code_root, code_receipt),
    "operational_bundle": bundle(ops_root, ops_receipt),
    "slurm_test_only": test_result,
    "jobs": [
        {"role": "train", "job_id": train, "partition": "normal", "nodes": 16, "updates": [int(start), int(end)]},
        {"role": "supervisor", "job_id": supervisor, "partition": "debug", "nodes": 1, "dependency": f"afterany:{train}"},
    ],
    "initial_segment_id": int(segment),
}
fd, temporary = tempfile.mkstemp(prefix=".launch.", suffix=".partial", dir=os.path.dirname(out))
with os.fdopen(fd, "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, out)
print(json.dumps({"train": train, "supervisor": supervisor, "experiment": experiment}, sort_keys=True))
PY
trap - ERR
