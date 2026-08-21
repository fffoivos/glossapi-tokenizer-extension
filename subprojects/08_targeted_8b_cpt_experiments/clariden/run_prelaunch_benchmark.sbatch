#!/usr/bin/env bash
#SBATCH --job-name=h2g-prelaunch-bench
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=03:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_BENCHMARK_CONTRACT H2G_BENCHMARK_OUTPUT_ROOT H2G_MEGATRON_DIR H2G_MEGATRON_RECEIPT H2G_VAL_DATA_DIR H2G_ONLINE_VALIDATION_RECEIPT H2G_EXTRA_VALID_SETS H2G_NEW_GREEK_VALID_SETS; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == normal ]] || { echo "prelaunch benchmark requires normal" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
read_contract() { /usr/bin/python3.11 - "$H2G_BENCHMARK_CONTRACT" "$1" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
cursor=value
for key in sys.argv[2].split('.'):
    cursor=cursor[key]
print(cursor)
PY
}
kind=$(read_contract kind)
scale=$(read_contract scale)
profile_id=$(read_contract profile_id)
nodes=$(read_contract nodes)
tp=$(read_contract tensor_parallel)
microbatch=$(read_contract microbatch)
peak=$(read_contract peak_lr)
floor=$(read_contract floor_lr)
init_root=$(read_contract initialization_root)
data_path=$(read_contract phase_data_path)
cache_root=$(read_contract phase_cache_root)
cache_tree=$(read_contract phase_cache_tree_sha256)
phase2_data_path=$(read_contract phase2_data_path)
phase2_cache_root=$(read_contract phase2_cache_root)
phase2_cache_tree=$(read_contract phase2_cache_tree_sha256)
[[ "$nodes" == "${SLURM_NNODES:-0}" ]] || { echo "contract/live nodes drift" >&2; exit 2; }
preflight="$H2G_STAGE_ROOT/receipts/prelaunch_${kind}_${scale}_${profile_id}_${peak}_${SLURM_JOB_ID}.json"
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/verify_prelaunch_benchmark_contract.py" \
  --contract "$H2G_BENCHMARK_CONTRACT" --output-root "$H2G_BENCHMARK_OUTPUT_ROOT" \
  --megatron-root "$H2G_MEGATRON_DIR" --megatron-receipt "$H2G_MEGATRON_RECEIPT" \
  --validation-root "$H2G_VAL_DATA_DIR" --validation-receipt "$H2G_ONLINE_VALIDATION_RECEIPT" --output "$preflight"

export H2G_MODEL_SCALE="$scale" H2G_PHASE=1 H2G_PEAK_LR="$peak" H2G_FLOOR_LR="$floor"
export H2G_MICROBATCH_SIZE="$microbatch" H2G_TENSOR_PARALLEL_SIZE="$tp"
export H2G_PHASE_DATA_PATH="$data_path" H2G_PHASE_CACHE_ROOT="$cache_root"
# The cache tree hash, not the receipt path, is the phase guard authority. Phase
# 1 does not install the cursor guard, but keeping the same environment makes
# the benchmark command auditable against production.
export H2G_PHASE_CACHE_TREE_SHA256="$cache_tree"
export H2G_MEGATRON_DIR H2G_VAL_DATA_DIR H2G_EXTRA_VALID_SETS H2G_NEW_GREEK_VALID_SETS
export H2G_CODE_ROOT H2G_STAGE_ROOT
export TRAIN_CONFIG_OVERRIDE="$subproject/clariden/hard_h_to_g_training.env"
export SCRIPT_DIR_OVERRIDE="$H2G_CODE_ROOT/frozen_training_tools/bakeoff_training"
export ACCOUNT=a0140 PARTITION=normal NODES="$SLURM_NNODES" GPUS_PER_NODE=4 LAUNCH_MODE=torchrun
export TIME_LIMIT="${H2G_TIME_LIMIT:-03:00:00}" ARM=td FULL_CPT_LAUNCH_VERIFY=0

request_stop() {
  find "$H2G_BENCHMARK_OUTPUT_ROOT" -type d -name triggers -print0 2>/dev/null | while IFS= read -r -d '' trigger; do : > "$trigger/save"; : > "$trigger/exit"; done
}
trap request_stop USR1 TERM INT

run_one() {
  output=$1 start=$2 exit_update=$3 load=$4 save_interval=$5 eval_interval=$6
  [[ ! -e "$output" ]] || { echo "benchmark subrun exists: $output" >&2; exit 2; }
  mkdir -p "$(dirname "$output")"
  export OUTPUT_DIR="$output" INIT_CKPT="$load" H2G_START_UPDATE="$start" H2G_EXIT_UPDATE="$exit_update"
  export H2G_SAVE_INTERVAL_OVERRIDE="$save_interval" H2G_EVAL_INTERVAL_OVERRIDE="$eval_interval"
  started_epoch=$(date +%s)
  set +e
  bash "$SCRIPT_DIR_OVERRIDE/bakeoff_train.sbatch" > "$output.driver.partial" 2>&1
  launcher_rc=$?
  set -e
  if (( launcher_rc != 0 )); then
    env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 \
      "$subproject/scripts/workaround_accept_intentional_torchrun_teardown.py" \
      --log "$output.driver.partial" \
      --checkpoint-root "$output/checkpoints" \
      --expected-update "$exit_update" \
      --launcher-returncode "$launcher_rc" \
      --output "$output/intentional_torchrun_teardown.json"
  fi
  mv "$output.driver.partial" "$output/driver.out"
  completed_epoch=$(date +%s)
  printf '%s\n' "$((completed_epoch - started_epoch))" > "$output/wall_seconds.txt"
}

freeze_update_one_load_view() {
  source_root=$1 view_root=$2 update=${3:-1}
  printf -v iteration_dir 'iter_%07d' "$update"
  checkpoint="$source_root/$iteration_dir"
  [[ -d "$checkpoint" && -f "$checkpoint/.metadata" ]] || {
    echo "uninterrupted update-$update distributed checkpoint is incomplete: $checkpoint" >&2
    exit 2
  }
  [[ ! -e "$view_root" ]] || { echo "restart load view exists: $view_root" >&2; exit 2; }
  mkdir -p "$view_root"
  ln -s "$checkpoint" "$view_root/$iteration_dir"
  printf '%s\n' "$update" > "$view_root/latest_checkpointed_iteration.txt"
  chmod a-w "$view_root/latest_checkpointed_iteration.txt" "$view_root"
}

mkdir -p "$(dirname "$H2G_BENCHMARK_OUTPUT_ROOT")"
if [[ "$kind" == profile ]]; then
  run_one "$H2G_BENCHMARK_OUTPUT_ROOT/restart/uninterrupted" 0 2 "$init_root" 1 1000000
  freeze_update_one_load_view \
    "$H2G_BENCHMARK_OUTPUT_ROOT/restart/uninterrupted/checkpoints" \
    "$H2G_BENCHMARK_OUTPUT_ROOT/restart/load_update_1"
  run_one "$H2G_BENCHMARK_OUTPUT_ROOT/restart/resumed" 1 2 "$H2G_BENCHMARK_OUTPUT_ROOT/restart/load_update_1" 1 1000000
  export H2G_PHASE=2 H2G_PHASE_START_UPDATE_OVERRIDE=1 H2G_PRELAUNCH_PHASE2_RESTART_SMOKE=1
  export H2G_PHASE_DATA_PATH="$phase2_data_path" H2G_PHASE_CACHE_ROOT="$phase2_cache_root"
  export H2G_PHASE_CACHE_TREE_SHA256="$phase2_cache_tree"
  run_one "$H2G_BENCHMARK_OUTPUT_ROOT/restart/phase2_uninterrupted" 1 3 "$H2G_BENCHMARK_OUTPUT_ROOT/restart/load_update_1" 1 1000000
  freeze_update_one_load_view \
    "$H2G_BENCHMARK_OUTPUT_ROOT/restart/phase2_uninterrupted/checkpoints" \
    "$H2G_BENCHMARK_OUTPUT_ROOT/restart/load_update_2" 2
  run_one "$H2G_BENCHMARK_OUTPUT_ROOT/restart/phase2_resumed" 2 3 "$H2G_BENCHMARK_OUTPUT_ROOT/restart/load_update_2" 1 1000000
  unset H2G_PHASE_START_UPDATE_OVERRIDE H2G_PRELAUNCH_PHASE2_RESTART_SMOKE
  export H2G_PHASE=1 H2G_PHASE_DATA_PATH="$data_path" H2G_PHASE_CACHE_ROOT="$cache_root"
  export H2G_PHASE_CACHE_TREE_SHA256="$cache_tree"
  run_one "$H2G_BENCHMARK_OUTPUT_ROOT/throughput" 0 256 "$init_root" 119 25
elif [[ "$kind" == lr ]]; then
  run_one "$H2G_BENCHMARK_OUTPUT_ROOT/pilot" 0 238 "$init_root" 238 238
else
  echo "invalid benchmark kind: $kind" >&2; exit 2
fi
cp "$preflight" "$H2G_BENCHMARK_OUTPUT_ROOT/preflight.json"
printf '%s\n' "$SLURM_JOB_ID" > "$H2G_BENCHMARK_OUTPUT_ROOT/slurm_job_id.txt"
chmod -R a-w "$H2G_BENCHMARK_OUTPUT_ROOT"
