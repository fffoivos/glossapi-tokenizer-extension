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

RECIPE=${FULL8_RECIPE:-$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json}
PROFILES=${FULL8_PROFILES:-$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json}
PREQUEUED_MANIFEST="$FULL8_RUN_ROOT/submissions/prequeued_launch_graph.json"

readarray -t selected < <(python3 - "$FULL8_SELECTED_PROFILE" "$FULL8_LAUNCH_GATE" "$FULL8_INITIAL_MEGATRON" "$RECIPE" <<'PY'
import json,sys
from pathlib import Path
p=json.load(open(sys.argv[1])); g=json.load(open(sys.argv[2]))
if p.get("schema_version")!="apertus_full_8b_selected_execution_profile_v1" or p.get("status")!="frozen": raise SystemExit("selected profile is not frozen")
if g.get("schema_version")!="apertus_full_8b_launch_gate_v1" or g.get("status")!="passed": raise SystemExit("launch gate is not passed")
s=p["selection"]
if Path(g.get("initialization_checkpoint",{}).get("root","")).resolve() != Path(sys.argv[3]).resolve(): raise SystemExit("launch checkpoint/gate drift")
if Path(p.get("recipe",{}).get("path","")).resolve()!=Path(sys.argv[4]).resolve(): raise SystemExit("selected profile/derived recipe drift")
print(s["profile_id"]); print(s["nodes"]); print(":".join(map(str,s["segment_boundaries"])))
PY
)
profile_id=${selected[0]}; nodes=${selected[1]}; IFS=: read -r -a boundaries <<<"${selected[2]}"
(( ${#boundaries[@]} >= 2 )) || { echo "selected profile has no segments" >&2; exit 2; }

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

common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_RUN_ROOT=$FULL8_RUN_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_PRELAUNCH_ROOT=$FULL8_PRELAUNCH_ROOT,FULL8_SELECTED_PROFILE=$FULL8_SELECTED_PROFILE,FULL8_EXECUTION_PROFILE=$profile_id,FULL8_LAUNCH_GATE=$FULL8_LAUNCH_GATE,FULL8_RECIPE=$RECIPE,FULL8_PROFILES=$PROFILES,FULL8_PREQUEUED_MANIFEST=$PREQUEUED_MANIFEST"
records=()
previous_supervisor=""
segment_count=$((${#boundaries[@]} - 1))
for ((segment=0; segment<segment_count; segment++)); do
  start=${boundaries[$segment]}
  end=${boundaries[$((segment + 1))]}
  load="$FULL8_RUN_ROOT/checkpoints"
  [[ "$segment" -eq 0 ]] && load="$FULL8_INITIAL_MEGATRON"
  dependency=()
  [[ -n "$previous_supervisor" ]] && dependency=(--dependency="afterok:$previous_supervisor")
  train=$(submit "${dependency[@]}" --nodes="$nodes" --job-name="full8b_s${segment}a0" \
    --output="$FULL8_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_RUN_ROOT/logs/%x-%j.err" \
    --export="$common,FULL8_SEGMENT_ID=$segment,FULL8_ATTEMPT=0,FULL8_START_ITERATION=$start,FULL8_END_ITERATION=$end,FULL8_LOAD_CHECKPOINT=$load,FULL8_RECOVERY_MODE=0" \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")
  supervisor=$(submit --dependency="afterany:$train" --job-name="full8b_supervise_s${segment}a0" \
    --output="$FULL8_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_RUN_ROOT/logs/%x-%j.err" \
    --export="$common" \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/supervise_campaign.sbatch" "$segment" 0 "$start" "$train")
  iterations=$(python3 - "$RECIPE" "$start" "$end" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); start,end=map(int,sys.argv[2:])
print(":".join(str(i) for i in r["evaluation"]["greekmmlu"]["checkpoint_updates"] if start < int(i) <= end))
PY
)
  [[ -n "$iterations" ]] || { echo "segment $segment has no GreekMMLU milestone" >&2; exit 2; }
  queue_receipt="$FULL8_RUN_ROOT/evaluation_queues/segment_${segment}.json"
  evaluation=$(submit --dependency="afterok:$supervisor" --job-name="full8b_evalq_s${segment}" \
    --output="$FULL8_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_RUN_ROOT/logs/%x-%j.err" \
    --export="$common,FULL8_TRAIN_JOB_ID=$train,FULL8_EVALUATION_ITERATIONS=$iterations,FULL8_EVALUATION_QUEUE_RECEIPT=$queue_receipt" \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_evaluation_queue.sbatch")
  records+=("$segment|$start|$end|$train|$supervisor|$evaluation|$iterations")
  previous_supervisor=$supervisor
done

training_receipt="$FULL8_RUN_ROOT/training_completion_receipt.json"
finalizer=$(submit --dependency="afterok:$previous_supervisor" --job-name=full8b_evidence_finalizer \
  --output="$FULL8_RUN_ROOT/logs/%x-%j.out" --error="$FULL8_RUN_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_TRAINING_RECEIPT=$training_receipt" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_campaign.sbatch")

graph=$(python3 - "$PREQUEUED_MANIFEST" "$profile_id" "$nodes" "$finalizer" "${records[@]}" <<'PY'
import datetime,json,os,sys,tempfile
out,profile,nodes,finalizer,*records=sys.argv[1:]
segments=[]
for raw in records:
    segment,start,end,train,supervisor,evaluation,iterations=raw.split("|",6)
    segments.append({
        "segment_id":int(segment),"start":int(start),"end":int(end),
        "train_job":train,"supervisor_job":supervisor,
        "evaluation_queue_job":evaluation,
        "greekmmlu_iterations":[int(value) for value in iterations.split(":") if value],
    })
payload={
    "schema_version":"apertus_full_8b_prequeued_launch_graph_v1",
    "status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "profile_id":profile,"nodes_per_training_segment":int(nodes),"segments":segments,
    "evidence_finalizer_job":finalizer,
    "continuation":"all happy-path training allocations are prequeued; each starts only after the preceding receipt gate",
    "failure_policy":"cancel only the rejected suffix and switch to the receipt-gated recovery chain",
}
if os.environ.get("DRY_RUN","1")=="0":
    fd,tmp=tempfile.mkstemp(prefix=".prequeued.",suffix=".partial",dir=os.path.dirname(out))
    with os.fdopen(fd,"w") as handle:
        json.dump(payload,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp,out)
print(json.dumps(payload,separators=(",",":")))
PY
)
if [[ "$DRY_RUN" == 0 ]]; then
  cp "$PREQUEUED_MANIFEST" "$FULL8_RUN_ROOT/submissions/launch_graph.json"
fi
printf '%s\n' "$graph"
