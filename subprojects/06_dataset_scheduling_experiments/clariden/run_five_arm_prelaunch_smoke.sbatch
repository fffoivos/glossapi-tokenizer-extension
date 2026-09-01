#!/usr/bin/env bash
#SBATCH --job-name=mini_5arm_smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --exclusive
#SBATCH --time=02:00:00

set -euo pipefail
: "${SCIENTIFIC_BUNDLE:?}" "${EFFICIENCY_BUNDLE:?}" "${MEGATRON_DIR:?}"
: "${TOKENIZER_DIR:?}" "${SCHEDULE_MANIFEST:?}" "${VALIDATION_MANIFEST:?}"
: "${TOKEN_BYTE_LENGTHS:?}" "${INITIAL_CHECKPOINT_ROOT:?}" "${LR_SELECTION_RECEIPT:?}" "${SMOKE_ROOT:?}"
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
[[ ! -e "$SMOKE_ROOT" ]] || { echo "refusing to replace $SMOKE_ROOT" >&2; exit 2; }
mkdir -p "$SMOKE_ROOT"
readarray -t lr < <("$HOST_PYTHON" - "$LR_SELECTION_RECEIPT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d["status"] == "frozen" and d["same_lr_for_all_five_arms"]
print(d["selected_peak_lr"]); print(d["selected_min_lr"])
PY
)
PEAK_LR=${lr[0]}; MIN_LR=${lr[1]}
mapfile -t all_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ "${#all_nodes[@]}" -eq 20 ]] || exit 2
arms=(D0_mixed D1_hard_h_to_g D2_hard_g_to_h D3_gradual_h_to_g D4_gradual_g_to_h)

run_phase() {
  local phase=$1 endpoint=$2 load_mode=$3
  local pids=() arm index offset nodes master output load_root
  for index in "${!arms[@]}"; do
    arm=${arms[$index]}; offset=$((index * 4)); nodes=$(IFS=,; echo "${all_nodes[*]:offset:4}"); master=${all_nodes[$offset]}
    output="$SMOKE_ROOT/$phase/$arm"; mkdir -p "$output"
    if [[ "$load_mode" == initial ]]; then load_root="$INITIAL_CHECKPOINT_ROOT"; else load_root="$SMOKE_ROOT/initial_to_64/$arm/checkpoints"; fi
    (
      set +e
      env SCIENTIFIC_BUNDLE="$SCIENTIFIC_BUNDLE" EFFICIENCY_BUNDLE="$EFFICIENCY_BUNDLE" MEGATRON_DIR="$MEGATRON_DIR" \
        TOKENIZER_DIR="$TOKENIZER_DIR" SCHEDULE_MANIFEST="$SCHEDULE_MANIFEST" VALIDATION_MANIFEST="$VALIDATION_MANIFEST" \
        TOKEN_BYTE_LENGTHS="$TOKEN_BYTE_LENGTHS" ARM_ID="$arm" ARM_NODELIST="$nodes" MASTER_ADDR="$master" MASTER_PORT="$((29700 + index))" \
        LOAD_ROOT="$load_root" ARM_OUTPUT_ROOT="$output" END_ITERATION="$endpoint" PEAK_LR="$PEAK_LR" MIN_LR="$MIN_LR" \
        bash "$SCIENTIFIC_BUNDLE/clariden/run_prelaunch_smoke_arm.sh" > "$output/driver.log" 2>&1
      rc=$?
      "$HOST_PYTHON" - "$output/process_state.json" "$rc" <<'PY'
import json,sys
open(sys.argv[1],"w").write(json.dumps({"exit_code":int(sys.argv[2])},indent=2)+"\n")
PY
      exit "$rc"
    ) &
    pids+=("$!")
  done
  local failure=0
  for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
  return "$failure"
}
run_phase initial_to_64 64 initial
run_phase resume_64_to_128 128 resume
"$HOST_PYTHON" "$SCIENTIFIC_BUNDLE/training/evaluate_five_arm_prelaunch_smoke.py" \
  --run-root "$SMOKE_ROOT" --validation-manifest "$VALIDATION_MANIFEST" \
  --output "$SMOKE_ROOT/five_arm_prelaunch_smoke_receipt.json"
