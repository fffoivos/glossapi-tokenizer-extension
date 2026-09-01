#!/usr/bin/env bash
#SBATCH --job-name=mini-cpt-five-arm
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"
: "${SEGMENT_ID:?set segment ID 0 or 1}"
: "${RUN_ROOT:?set immutable campaign run root}"
: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
SEGMENT_ATTEMPT=${SEGMENT_ATTEMPT:-0}

SEGMENT_ROOT="$RUN_ROOT/segments/segment_${SEGMENT_ID}/attempt_${SEGMENT_ATTEMPT}"
PREFLIGHT="$SEGMENT_ROOT/segment_preflight.json"
mkdir -p "$SEGMENT_ROOT"
preflight_args=(
  --campaign-manifest "$CAMPAIGN_MANIFEST" --segment-id "$SEGMENT_ID"
  --run-root "$RUN_ROOT" --output "$PREFLIGHT"
)
if [[ -n "${RESUME_RECEIPT:-}" ]]; then
  preflight_args+=(--resume-receipt "$RESUME_RECEIPT")
fi
if [[ -n "${RECOVERY_START:-}" ]]; then
  preflight_args+=(--recovery-start "$RECOVERY_START")
fi
if [[ -n "${RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT:-}" ]]; then
  preflight_args+=(
    --runtime-scientific-bundle "$SCIENTIFIC_BUNDLE"
    --runtime-scientific-bundle-receipt "$RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT"
  )
fi
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/production/preflight_segment.py" "${preflight_args[@]}"

mapfile -t all_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#all_nodes[@]}" -eq 20 ]] || { echo "expected exactly 20 nodes" >&2; exit 2; }
arm_ids=(D0_mixed D1_hard_h_to_g D2_hard_g_to_h D3_gradual_h_to_g D4_gradual_g_to_h)
declare -A pid_to_arm=()
declare -A exit_codes=()

request_common_stop() {
  local arm
  for arm in "${arm_ids[@]}"; do
    mkdir -p "$SEGMENT_ROOT/$arm/triggers"
    touch "$SEGMENT_ROOT/$arm/triggers/save" "$SEGMENT_ROOT/$arm/triggers/exit"
  done
}
trap request_common_stop USR1 TERM INT

for arm_index in "${!arm_ids[@]}"; do
  arm=${arm_ids[$arm_index]}
  offset=$((arm_index * 4))
  node_csv=$(IFS=,; echo "${all_nodes[*]:offset:4}")
  master=${all_nodes[$offset]}
  output="$SEGMENT_ROOT/$arm"
  mkdir -p "$output"
  env CAMPAIGN_MANIFEST="$CAMPAIGN_MANIFEST" SEGMENT_PREFLIGHT="$PREFLIGHT" \
    ARM_ID="$arm" ARM_NODELIST="$node_csv" ARM_OUTPUT_ROOT="$output" \
    MASTER_ADDR="$master" MASTER_PORT="$((29600 + arm_index))" \
    UENV_IMAGE="${UENV_IMAGE:-pytorch/v2.9.1:v2}" \
    bash "$SCIENTIFIC_BUNDLE/clariden/run_production_arm_segment.sh" \
      >"$output/driver.out" 2>"$output/driver.err" &
  pid=$!
  pid_to_arm[$pid]=$arm
done

failure=0
while ((${#pid_to_arm[@]})); do
  for pid in "${!pid_to_arm[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      arm=${pid_to_arm[$pid]}
      if wait "$pid"; then rc=0; else rc=$?; fi
      exit_codes[$arm]=$rc
      unset 'pid_to_arm[$pid]'
      if ((rc != 0 && failure == 0)); then
        failure=1
        request_common_stop
      fi
    fi
  done
  ((${#pid_to_arm[@]})) && sleep 5
done

code_args=()
for arm in "${arm_ids[@]}"; do code_args+=("${exit_codes[$arm]:-255}"); done
"$HOST_PYTHON" - "$SEGMENT_ROOT/segment_state.json" "$SEGMENT_ID" "$failure" \
  "${arm_ids[@]}" -- "${code_args[@]}" <<'PY'
import datetime,json,os,sys,tempfile
out,segment,failure,*rest=sys.argv[1:]
split=rest.index("--"); arms=rest[:split]; codes=rest[split+1:]
payload={
 "schema_version":"apertus_mini_segment_execution_state_v1",
 "status":"completed" if int(failure)==0 and all(int(x)==0 for x in codes) else "failed",
 "completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "segment_id":int(segment),
 "arms":{arm:{"exit_code":int(code)} for arm,code in zip(arms,codes)},
}
directory=os.path.dirname(out); fd,tmp=tempfile.mkstemp(prefix=".segment.",suffix=".partial",dir=directory)
with os.fdopen(fd,"w") as f: json.dump(payload,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
((failure == 0))
