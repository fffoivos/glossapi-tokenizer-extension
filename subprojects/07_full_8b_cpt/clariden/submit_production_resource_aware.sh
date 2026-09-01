#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_STAGE_ROOT:?set frozen data root}"
: "${FULL8_RUN_ROOT:?set new production run root}"
: "${FULL8_INITIAL_MEGATRON:?set verified TP2 initialization root}"
: "${FULL8_PRELAUNCH_ROOT:?set completed prelaunch root}"
: "${FULL8_SELECTED_PROFILE:?set successor-bound selected profile}"
: "${FULL8_LAUNCH_GATE:?set completed launch gate}"
: "${FULL8_OPERATIONAL_LAUNCH_GATE:?set completed operational launch gate}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set pinned Clariden leaf switch}"
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || {
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
}

RECIPE=${FULL8_RECIPE:-$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json}
PROFILES=${FULL8_PROFILES:-$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json}
MANIFEST="$FULL8_RUN_ROOT/submissions/resource_aware_launch_graph.json"

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

readarray -t selected < <(/usr/bin/python3.11 - "$FULL8_SELECTED_PROFILE" "$FULL8_LAUNCH_GATE" "$RECIPE" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
if p.get("schema_version")!="apertus_full_8b_selected_execution_profile_v1" or p.get("status")!="frozen": raise SystemExit("selected profile is not frozen")
if g.get("schema_version")!="apertus_full_8b_launch_gate_v1" or g.get("status")!="passed": raise SystemExit("launch gate is not passed")
if p["selection"] != g["selected_profile"]: raise SystemExit("launch gate/selected profile drift")
if Path(p["recipe"]["path"]).resolve()!=Path(sys.argv[3]).resolve(): raise SystemExit("successor recipe binding drift")
s=p["selection"]; b=list(map(int,s["segment_boundaries"]))
if s["profile_id"]!="dp32_16node" or s["nodes"]!=16 or len(b)!=6: raise SystemExit("production geometry drift")
print(s["profile_id"]); print(s["nodes"]); print(b[0]); print(b[1])
PY
)
/usr/bin/python3.11 - "$FULL8_OPERATIONAL_LAUNCH_GATE" "$FULL8_CODE_ROOT" "$FULL8_OPS_ROOT" <<'PY'
import json,os,sys
d=json.load(open(sys.argv[1]))
if d.get("schema_version")!="apertus_full_8b_operational_launch_gate_v1" or d.get("status")!="passed": raise SystemExit("operational launch gate is not passed")
if os.path.realpath(d.get("scientific_root",""))!=os.path.realpath(sys.argv[2]): raise SystemExit("operational/scientific root drift")
if os.path.realpath(d.get("operational_root",""))!=os.path.realpath(sys.argv[3]): raise SystemExit("operational root drift")
PY
profile_id=${selected[0]}; nodes=${selected[1]}; start=${selected[2]}; end=${selected[3]}
[[ "$start" == 0 ]] || { echo "segment 0 must start at update 0" >&2; exit 2; }
train_exclude=$("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$FULL8_TRAIN_LEAF_SWITCH" "$nodes")
[[ -n "$train_exclude" ]] || { echo "empty leaf exclusion" >&2; exit 2; }

if [[ "$DRY_RUN" == 0 ]]; then
  [[ "${CONFIRM_GPU_LAUNCH:-}" == APERTUS8B_FULL_MIXED_CPT ]] || {
    echo "set CONFIRM_GPU_LAUNCH=APERTUS8B_FULL_MIXED_CPT" >&2
    exit 2
  }
  [[ ! -e "$FULL8_RUN_ROOT" ]] || { echo "run root already exists" >&2; exit 2; }
  mkdir -p "$FULL8_RUN_ROOT/logs" "$FULL8_RUN_ROOT/submissions" \
    "$FULL8_RUN_ROOT/orchestration/events"
fi

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    printf 'DRY_JOB_%s\n' "$(printf '%s\0' "$@" | cksum | awk '{print $1}')"
  else
    sbatch --uenv-passthrough=ignore --parsable "$@"
  fi
}

common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_OPS_ROOT=$FULL8_OPS_ROOT,FULL8_OPS_BUNDLE_RECEIPT=$FULL8_OPS_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_RUN_ROOT=$FULL8_RUN_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_PRELAUNCH_ROOT=$FULL8_PRELAUNCH_ROOT,FULL8_SELECTED_PROFILE=$FULL8_SELECTED_PROFILE,FULL8_EXECUTION_PROFILE=$profile_id,FULL8_LAUNCH_GATE=$FULL8_LAUNCH_GATE,FULL8_RECIPE=$RECIPE,FULL8_PROFILES=$PROFILES,FULL8_TRAIN_LEAF_SWITCH=$FULL8_TRAIN_LEAF_SWITCH"

train=$(submit --partition=normal --time=12:00:00 --switches=1 \
  --exclude="$train_exclude" --nodes="$nodes" --job-name=full8b_s0a0 \
  --output="$FULL8_RUN_ROOT/logs/%x-%j.out" \
  --error="$FULL8_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_SEGMENT_ID=0,FULL8_ATTEMPT=0,FULL8_START_ITERATION=$start,FULL8_END_ITERATION=$end,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON,FULL8_RECOVERY_MODE=0" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")
supervisor=$(submit --partition=debug --time=00:20:00 --nodes=1 \
  --dependency="afterany:$train" --job-name=full8b_supervise_s0a0 \
  --output="$FULL8_RUN_ROOT/logs/%x-%j.out" \
  --error="$FULL8_RUN_ROOT/logs/%x-%j.err" --export="$common" \
  "$FULL8_OPS_ROOT/clariden/supervise_campaign_resource_aware.sbatch" \
  0 0 "$start" "$train")

/usr/bin/python3.11 - "$MANIFEST" "$DRY_RUN" "$train" "$supervisor" "$start" "$end" "$nodes" "$profile_id" \
  "$FULL8_CODE_ROOT" "$FULL8_CODE_BUNDLE_RECEIPT" "$FULL8_OPS_ROOT" "$FULL8_OPS_BUNDLE_RECEIPT" "$FULL8_OPERATIONAL_LAUNCH_GATE" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,dry,train,supervisor,start,end,nodes,profile,code_root,code_receipt,ops_root,ops_receipt,ops_gate=sys.argv[1:]
def binding(root,receipt):
 raw=open(receipt,"rb").read(); value=json.loads(raw)
 if os.path.realpath(value["root"])!=os.path.realpath(root): raise SystemExit("bundle root drift")
 return {"root":os.path.realpath(root),"receipt":os.path.realpath(receipt),"receipt_sha256":hashlib.sha256(raw).hexdigest(),"tree_sha256":value["tree_sha256"]}
def file_binding(path):
 raw=open(path,"rb").read()
 return {"path":os.path.realpath(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
value={
 "schema_version":"apertus_full_8b_resource_aware_launch_graph_v1",
 "status":"dry_run" if dry=="1" else "submitted",
 "created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "profile_id":profile,"policy":"dynamic_normal_train_serial_debug_control_v1",
 "scientific_bundle":binding(code_root,code_receipt),
 "operational_bundle":binding(ops_root,ops_receipt),
 "operational_launch_gate":file_binding(ops_gate),
 "jobs":[
  {"role":"train","job_id":train,"partition":"normal","nodes":int(nodes),"time":"12:00:00","updates":[int(start),int(end)]},
  {"role":"supervisor","job_id":supervisor,"partition":"debug","nodes":1,"time":"00:20:00","dependency":f"afterany:{train}"},
 ],
 "continuation":"each supervisor submits the next normal train and one serial debug evaluation; the final evaluation submits the next supervisor",
}
if dry=="0":
 fd,tmp=tempfile.mkstemp(prefix=".launch.",suffix=".partial",dir=os.path.dirname(out))
 with os.fdopen(fd,"w") as f:
  json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
 os.replace(tmp,out)
print(json.dumps(value,separators=(",",":")))
PY
