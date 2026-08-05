#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set executing code-bundle receipt}"
: "${FULL8_STAGE_ROOT:?set frozen data root}"
: "${FULL8_RUN_ROOT:?set new production run root}"
: "${FULL8_INITIAL_MEGATRON:?set verified TP2 initialization root}"
: "${FULL8_PRELAUNCH_ROOT:?set completed prelaunch root}"
: "${FULL8_SELECTED_PROFILE:?set benchmark-selected execution profile receipt}"
: "${FULL8_LAUNCH_GATE:?set completed authoritative launch gate}"
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }

readarray -t selected < <(python3 - "$FULL8_SELECTED_PROFILE" "$FULL8_LAUNCH_GATE" "$FULL8_INITIAL_MEGATRON" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
if p.get("schema_version")!="apertus_full_8b_selected_execution_profile_v1" or p.get("status")!="frozen": raise SystemExit("selected profile is not frozen")
if g.get("schema_version")!="apertus_full_8b_launch_gate_v1" or g.get("status")!="passed": raise SystemExit("launch gate is not passed")
s=p["selection"]
if Path(g.get("initialization_checkpoint",{}).get("root","")).resolve() != Path(sys.argv[3]).resolve(): raise SystemExit("launch checkpoint/gate drift")
print(s["profile_id"]); print(s["nodes"]); print(s["segment_boundaries"][1])
PY
)
profile_id=${selected[0]}; nodes=${selected[1]}; first_end=${selected[2]}

if [[ "$DRY_RUN" == 0 ]]; then
  [[ "${CONFIRM_GPU_LAUNCH:-}" == "APERTUS8B_FULL_MIXED_CPT" ]] || {
    echo "set CONFIRM_GPU_LAUNCH=APERTUS8B_FULL_MIXED_CPT" >&2; exit 2;
  }
  [[ ! -e "$FULL8_RUN_ROOT" ]] || { echo "run root already exists" >&2; exit 2; }
  mkdir -p "$FULL8_RUN_ROOT/logs" "$FULL8_RUN_ROOT/submissions" "$FULL8_RUN_ROOT/orchestration/events"
fi

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    printf 'DRY_JOB_%s\n' "$(printf '%s\0' "$@" | cksum | awk '{print $1}')"
  else
    sbatch --parsable "$@"
  fi
}

common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_RUN_ROOT=$FULL8_RUN_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_PRELAUNCH_ROOT=$FULL8_PRELAUNCH_ROOT,FULL8_SELECTED_PROFILE=$FULL8_SELECTED_PROFILE,FULL8_EXECUTION_PROFILE=$profile_id,FULL8_LAUNCH_GATE=$FULL8_LAUNCH_GATE"
train=$(submit --nodes="$nodes" --job-name=full8b_s0a0 \
  --output="$FULL8_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_SEGMENT_ID=0,FULL8_ATTEMPT=0,FULL8_START_ITERATION=0,FULL8_END_ITERATION=$first_end,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON,FULL8_RECOVERY_MODE=0" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")
supervisor=$(submit --dependency="afterany:$train" --job-name=full8b_supervise_s0a0 \
  --output="$FULL8_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_RUN_ROOT/logs/%x-%j.err" \
  --export="$common" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/supervise_campaign.sbatch" 0 0 0 "$train")

graph=$(python3 - "$train" "$supervisor" "$profile_id" "$nodes" "$first_end" <<'PY'
import json,sys
print(json.dumps({"initial_train_job":sys.argv[1],"initial_supervisor_job":sys.argv[2],"profile_id":sys.argv[3],"nodes":int(sys.argv[4]),"first_segment":{"start":0,"end":int(sys.argv[5])},"continuation":"receipt-gated segment supervisor"},separators=(",",":")))
PY
)
if [[ "$DRY_RUN" == 0 ]]; then
  python3 - "$FULL8_RUN_ROOT/submissions/launch_graph.json" "$graph" <<'PY'
import datetime,json,os,sys,tempfile
out,raw=sys.argv[1:]; payload={"schema_version":"apertus_full_8b_launch_graph_v2","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"jobs":json.loads(raw)}
fd,tmp=tempfile.mkstemp(prefix=".launch.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
fi
printf '%s\n' "$graph"
