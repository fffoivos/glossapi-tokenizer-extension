#!/usr/bin/env bash
# Read-only status snapshot for a submitted five-arm production campaign.
set -euo pipefail

: "${RUN_ROOT:?set the immutable campaign run root}"
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
EVALUATION_NAMESPACE=${EVALUATION_NAMESPACE:-fp32_v1}
evaluation_watch_root="$RUN_ROOT/evaluation_watch_${EVALUATION_NAMESPACE}"
receipt="$RUN_ROOT/submission_receipt.json"
[[ -f "$receipt" ]] || { echo "missing submission receipt: $receipt" >&2; exit 2; }

readarray -t contract < <("$HOST_PYTHON" - "$receipt" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
print(r["run_tag"])
for name, job_id in r["jobs"].items():
    print(f"{name}\t{job_id}")
PY
)
run_tag=${contract[0]}

readarray -t recovery_jobs < <("$HOST_PYTHON" - "$RUN_ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]) / "orchestration" / "recoveries"
seen=set()
for path in sorted(root.rglob("*.json")):
    value=json.load(path.open())
    for key in ("replacement_watcher_job_id", "replacement_supervisor_job_id"):
        job_id=str(value.get(key, ""))
        if job_id and job_id not in seen:
            print(f"{path.stem}:{key}\t{job_id}")
            seen.add(job_id)
PY
)

readarray -t evaluation_jobs < <("$HOST_PYTHON" - "$evaluation_watch_root" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for path in sorted(root.glob("segment_*/iteration_*.json")):
    value=json.load(path.open())
    attempts=value.get("attempts", [])
    if not attempts:
        continue
    latest=attempts[-1]
    segment=path.parent.name.removeprefix("segment_")
    iteration=int(value["iteration"])
    print(
        f"greekmmlu_s{segment}_i{iteration:07d}_a{latest['attempt']}\t"
        f"{latest['job_id']}\t{value.get('status', 'unknown')}"
    )
PY
)

echo "observed_at_utc=$(date -u +%FT%TZ)"
echo "run_tag=$run_tag"
echo "run_root=$RUN_ROOT"
echo "evaluation_namespace=$EVALUATION_NAMESPACE"
echo "submission_jobs"
for row in "${contract[@]:1}"; do
  IFS=$'\t' read -r name job_id <<<"$row"
  printf '%s\t%s\n' "$name" "$job_id"
done

echo "recovery_jobs"
if ((${#recovery_jobs[@]})); then
  printf '%s\n' "${recovery_jobs[@]}"
else
  echo none
fi

echo "evaluation_watch_jobs"
if ((${#evaluation_jobs[@]})); then
  printf '%s\n' "${evaluation_jobs[@]}"
else
  echo none
fi

echo "slurm_queue"
echo 'JOBID|NAME|STATE|TIME|START_TIME|NODES|NODELIST(REASON)'
job_csv=$(
  {
    printf '%s\n' "${contract[@]:1}" | cut -f2
    if ((${#recovery_jobs[@]})); then
      printf '%s\n' "${recovery_jobs[@]}" | cut -f2
    fi
    if ((${#evaluation_jobs[@]})); then
      printf '%s\n' "${evaluation_jobs[@]}" | cut -f2
    fi
  } | awk 'NF && !seen[$0]++' | paste -sd, -
)
squeue -j "$job_csv" -h -o '%.18i|%.44j|%.10T|%.12M|%.19S|%.6D|%R' || true

echo "slurm_accounting"
sacct -j "$job_csv" --format=JobID,JobName%44,State,Elapsed,Start,End,ExitCode -P || true

echo "receipts"
find "$RUN_ROOT" -maxdepth 8 -type f \
  \( -name '*receipt*.json' -o -name 'segment_state.json' -o -name 'summary.json' -o -name 'latest.json' \) \
  -print 2>/dev/null | sort

echo "greekmmlu_receipt_audit"
"$HOST_PYTHON" - "$evaluation_watch_root" "$EVALUATION_NAMESPACE" <<'PY'
import hashlib,json,math,sys
from pathlib import Path

arms=("D0_mixed","D1_hard_h_to_g","D2_hard_g_to_h","D3_gradual_h_to_g","D4_gradual_g_to_h")
root=Path(sys.argv[1])
namespace=sys.argv[2]
states={"waiting_for_checkpoint":0,"submitted":0,"completed":0,"failed":0,"other":0}
valid=0
invalid=[]
for state_path in sorted(root.glob("segment_*/iteration_*.json")):
    state=json.load(state_path.open())
    status=state.get("status", "other")
    states[status if status in states else "other"] += 1
    if status != "completed":
        continue
    iteration=int(state["iteration"])
    for arm in arms:
        try:
            receipt_path=Path(state["receipts"][arm])
            receipt=json.load(receipt_path.open())
            pipeline=json.load((receipt_path.parent / "pipeline_state.json").open())
            metrics=receipt["metrics"]
            clean=metrics["decontaminated"]
            export_path=Path(receipt["checkpoint"]["export_receipt_path"])
            values=(metrics["accuracy"],metrics["choice_nll"],metrics["correct_answer_bpb"],clean["accuracy"],clean["choice_nll"],clean["correct_answer_bpb"])
            assert receipt["schema_version"] == "exact_checkpoint_native_greekmmlu_receipt_v1"
            assert receipt["status"] == "completed"
            assert receipt["evaluation_namespace"] == namespace
            assert receipt["evaluator"]["dtype"] == "float32"
            assert int(receipt["checkpoint"]["iteration"]) == iteration
            assert int(metrics["n"]) == 16632 and int(clean["n"]) > 0
            assert all(math.isfinite(float(value)) for value in values)
            assert export_path.is_file()
            assert hashlib.sha256(export_path.read_bytes()).hexdigest() == receipt["checkpoint"]["export_receipt_sha256"]
            assert pipeline["status"] == "complete" and pipeline["arm_id"] == arm
            assert int(pipeline["source_iteration"]) == iteration
            assert Path(pipeline["receipt"]).resolve() == receipt_path.resolve()
            valid += 1
        except Exception as error:
            invalid.append(f"{arm}@{iteration}:{type(error).__name__}:{error}")
print("states=" + ",".join(f"{key}:{value}" for key,value in states.items()))
print(f"valid_bindings={valid} invalid_bindings={len(invalid)}")
for row in invalid[:20]:
    print("invalid=" + row)
PY

echo "source_validation_audit"
"$HOST_PYTHON" - "$RUN_ROOT" <<'PY'
import json,math,re,sys
from pathlib import Path

root=Path(sys.argv[1])
pattern=re.compile(
    r"validation loss at iteration (\d+) \[([^]]+)\].*?"
    r"lm loss value: ([-+0-9.Ee]+)"
)
rows={}
nonfinite=[]
for driver in sorted(root.glob("segments/segment_*/attempt_*/D*/driver.out")):
    arm=driver.parent.name
    segment_attempt="/".join(driver.parts[-4:-2])
    key=f"{segment_attempt}/{arm}"
    points={}
    for line in driver.read_text(encoding="utf-8",errors="replace").splitlines():
        match=pattern.search(line)
        if match is None:
            continue
        iteration,panel,value=int(match.group(1)),match.group(2),float(match.group(3))
        points.setdefault(iteration,set()).add(panel)
        if not math.isfinite(value):
            nonfinite.append(f"{key}@{iteration}:{panel}={value}")
    complete=sorted(iteration for iteration,panels in points.items() if len(panels)==13)
    rows[key]={
        "complete_points":len(complete),
        "latest_complete_iteration":complete[-1] if complete else None,
        "panel_bindings":sum(len(panels) for panels in points.values()),
        "incomplete_points":{
            str(iteration):len(panels)
            for iteration,panels in sorted(points.items())
            if len(panels)!=13
        },
    }
print(json.dumps({"arms":rows,"nonfinite":nonfinite},sort_keys=True))
PY

echo "arm_health"
for segment_root in "$RUN_ROOT"/segments/segment_*/attempt_*; do
  [[ -d "$segment_root" ]] || continue
  for arm_root in "$segment_root"/D*; do
    [[ -d "$arm_root" ]] || continue
    arm=$(basename "$arm_root")
    latest_marker="$arm_root/checkpoints/latest_checkpointed_iteration.txt"
    latest_checkpoint=none
    [[ -f "$latest_marker" ]] && latest_checkpoint=$(tr -d '[:space:]' <"$latest_marker")
    driver="$arm_root/driver.out"
    latest_iteration=none
    if [[ -f "$driver" ]]; then
      latest_iteration=$(grep -E 'iteration[[:space:]]+[0-9]+/' "$driver" | tail -1 \
        | sed -E 's/.*iteration[[:space:]]+([0-9]+)\/.*/\1/' || true)
      [[ -n "$latest_iteration" ]] || latest_iteration=none
    fi
    error_count=0
    logs=()
    [[ -f "$arm_root/driver.out" ]] && logs+=("$arm_root/driver.out")
    [[ -f "$arm_root/driver.err" ]] && logs+=("$arm_root/driver.err")
    if ((${#logs[@]})); then
      error_count=$(grep -Eih 'Traceback|CUDA out of memory|NCCL.*(error|failed)|(lm loss|grad norm)[^|]*(nan|inf)' \
        "${logs[@]}" | wc -l | tr -d '[:space:]' || true)
      [[ -n "$error_count" ]] || error_count=0
    fi
    printf '%s\t%s\tlatest_iteration=%s\tlatest_checkpoint=%s\terror_matches=%s\n' \
      "$(basename "$(dirname "$segment_root")")/$(basename "$segment_root")" \
      "$arm" "$latest_iteration" "$latest_checkpoint" "$error_count"
  done
done
