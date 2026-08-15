#!/usr/bin/env bash
#SBATCH --job-name=target8-restart-smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --switches=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --exclusive
#SBATCH --mem=450G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_EXPERIMENT:?set A or B}"
: "${TARGET8_CODE_ROOT:?set immutable scientific code root}"
: "${TARGET8_CODE_RECEIPT:?set scientific bundle receipt}"
: "${TARGET8_STAGE_ROOT:?set common targeted preparation root}"
: "${TARGET8_SMOKE_ROOT:?set new restart-smoke root}"
: "${TARGET8_INITIAL_CHECKPOINT_ROOT:?set A init root or B parent checkpoint root}"
[[ "$TARGET8_EXPERIMENT" == A || "$TARGET8_EXPERIMENT" == B ]] || { echo "TARGET8_EXPERIMENT must be A or B" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == normal && "${SLURM_NNODES:-0}" == 16 ]] || {
  echo "targeted restart smoke requires one 16-node normal allocation" >&2; exit 2;
}
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific

name=$(printf '%s' "$TARGET8_EXPERIMENT" | tr '[:upper:]' '[:lower:]')
stage="$TARGET8_STAGE_ROOT/experiment_$name"
recipe="$stage/contracts/recipe_8b_full_mixed.sanitized.json"
profiles="$stage/contracts/execution_profiles.sanitized.json"
selected="$stage/contracts/selected_execution_profile.json"
schedule="$stage/schedules/schedule_manifest.json"
assets="$stage/contracts/training_assets_receipt.json"
for path in "$recipe" "$profiles" "$selected" "$schedule" "$assets"; do
  [[ -s "$path" ]] || { echo "frozen smoke input missing: $path" >&2; exit 2; }
done
[[ -d "$TARGET8_SMOKE_ROOT/logs" ]] || { echo "submitted smoke root/log directory missing" >&2; exit 2; }
for path in "$TARGET8_SMOKE_ROOT/control" "$TARGET8_SMOKE_ROOT/resume" "$TARGET8_SMOKE_ROOT/restart_smoke_receipt.json"; do
  [[ ! -e "$path" ]] || { echo "immutable smoke output exists: $path" >&2; exit 2; }
done
mkdir "$TARGET8_SMOKE_ROOT/control" "$TARGET8_SMOKE_ROOT/resume"

if [[ "$TARGET8_EXPERIMENT" == A ]]; then
  base=0
  exact_initial=""
else
  base=9536
  exact_initial=9536
  expected=$(/usr/bin/python3.11 - "$recipe" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["initialization"]["targeted_resume"]["checkpoint_directory"])
PY
)
  [[ "$(realpath "$TARGET8_INITIAL_CHECKPOINT_ROOT/iter_0009536")" == "$(realpath "$expected")" ]] || {
    echo "B restart-smoke checkpoint authority drift" >&2; exit 2;
  }
fi
checkpoint=$((base + 1))
comparison=$((base + 2))
train="$TARGET8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"

request_smoke_stop() {
  set +e
  mkdir -p "$TARGET8_SMOKE_ROOT/control/triggers" "$TARGET8_SMOKE_ROOT/resume/triggers"
  touch "$TARGET8_SMOKE_ROOT/control/triggers/save" "$TARGET8_SMOKE_ROOT/control/triggers/exit"
  touch "$TARGET8_SMOKE_ROOT/resume/triggers/save" "$TARGET8_SMOKE_ROOT/resume/triggers/exit"
  set -e
}
trap request_smoke_stop USR1 TERM INT

run_train() {
  local run_root=$1 start=$2 end=$3 load_root=$4 save_iterations=$5 exact_iteration=$6
  env \
    FULL8_CODE_ROOT="$TARGET8_CODE_ROOT" FULL8_CODE_BUNDLE_RECEIPT="$TARGET8_CODE_RECEIPT" \
    FULL8_STAGE_ROOT="$stage" FULL8_RUN_ROOT="$run_root" \
    FULL8_RECIPE="$recipe" FULL8_PROFILES="$profiles" FULL8_EXECUTION_PROFILE=dp32_16node \
    FULL8_START_ITERATION="$start" FULL8_END_ITERATION="$end" FULL8_LOAD_CHECKPOINT="$load_root" \
    FULL8_BENCHMARK_MODE=1 FULL8_BENCHMARK_BASE_ITERATION="$base" \
    FULL8_BENCHMARK_SAVE_ITERATIONS="$save_iterations" FULL8_EXACT_LOAD_ITERATION="$exact_iteration" \
    bash "$train"
}

# Two updates without process interruption, with the real source-panel
# validation and synchronous-save boundary at `checkpoint`.
run_train "$TARGET8_SMOKE_ROOT/control" "$base" "$comparison" \
  "$TARGET8_INITIAL_CHECKPOINT_ROOT" "$checkpoint" "$exact_initial"
# Resume from the *exact checkpoint emitted by the uninterrupted control*.
# Recomputing the first update in a second process would fold independent
# cross-launch numerical drift into the restart test before the restart even
# occurs; it is not a valid isolation of checkpoint/load behavior.
run_train "$TARGET8_SMOKE_ROOT/resume" "$checkpoint" "$comparison" \
  "$TARGET8_SMOKE_ROOT/control/checkpoints" "" "$checkpoint"

uenv run pytorch/v2.9.1:v2 --view=default -- \
  /usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_targeted_restart_smoke.py" \
  --experiment "$TARGET8_EXPERIMENT" --code-root "$TARGET8_CODE_ROOT" \
  --code-bundle-receipt "$TARGET8_CODE_RECEIPT" \
  --recipe "$recipe" --profiles "$profiles" --selected-profile "$selected" \
  --schedule-manifest "$schedule" --validation-manifest "$stage/validation/validation_manifest.json" \
  --control-root "$TARGET8_SMOKE_ROOT/control" \
  --resume-root "$TARGET8_SMOKE_ROOT/resume" --base-iteration "$base" \
  --output "$TARGET8_SMOKE_ROOT/restart_smoke_receipt.json"
