#!/usr/bin/env bash
set -euo pipefail
: "${TARGET8_EXPERIMENT:?set A or B}"
: "${TARGET8_CODE_ROOT:?set immutable scientific code root}"
: "${TARGET8_CODE_RECEIPT:?set scientific bundle receipt}"
: "${TARGET8_STAGE_ROOT:?set common targeted preparation root}"
: "${TARGET8_SMOKE_ROOT:?set new restart-smoke root}"
: "${TARGET8_INITIAL_CHECKPOINT_ROOT:?set A init root or B parent checkpoint root}"
: "${TARGET8_LEAF_SWITCH:?set pinned Clariden leaf switch or auto}"
DRY_RUN=${DRY_RUN:-1}
[[ "$TARGET8_EXPERIMENT" == A || "$TARGET8_EXPERIMENT" == B ]] || { echo "TARGET8_EXPERIMENT must be A or B" >&2; exit 2; }
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
name=$(printf '%s' "$TARGET8_EXPERIMENT" | tr '[:upper:]' '[:lower:]')
stage="$TARGET8_STAGE_ROOT/experiment_$name"
for path in "$stage/contracts/training_assets_receipt.json" "$stage/contracts/selected_execution_profile.json"; do
  [[ -s "$path" ]] || { echo "distributed smoke submitted before debug asset freeze: $path" >&2; exit 2; }
done
exclude_args=()
if [[ "$TARGET8_LEAF_SWITCH" != auto ]]; then
  exclude=$(
    "$TARGET8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" \
      "$TARGET8_LEAF_SWITCH" 16
  )
  [[ -n "$exclude" ]] || { echo "empty leaf exclusion" >&2; exit 2; }
  exclude_args=(--exclude="$exclude")
fi
script="$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/run_targeted_restart_smoke.sbatch"
exports="ALL,TARGET8_EXPERIMENT=$TARGET8_EXPERIMENT,TARGET8_CODE_ROOT=$TARGET8_CODE_ROOT,TARGET8_CODE_RECEIPT=$TARGET8_CODE_RECEIPT,TARGET8_STAGE_ROOT=$TARGET8_STAGE_ROOT,TARGET8_SMOKE_ROOT=$TARGET8_SMOKE_ROOT,TARGET8_INITIAL_CHECKPOINT_ROOT=$TARGET8_INITIAL_CHECKPOINT_ROOT"
command=(sbatch --uenv-passthrough=ignore --parsable --partition=normal --nodes=16 --time=01:00:00 \
  --switches=1 "${exclude_args[@]}" --job-name="target8${name}-restart-smoke" \
  --output="$TARGET8_SMOKE_ROOT/logs/%x-%j.out" --error="$TARGET8_SMOKE_ROOT/logs/%x-%j.err" \
  --export="$exports" "$script")
test_result=$(sbatch --test-only "${command[@]:1}" 2>&1) || { echo "$test_result" >&2; exit 2; }
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'RESTART_SMOKE_TEST_ONLY_OK %s\n' "$test_result"
  { printf 'COMMAND'; printf ' %q' "${command[@]}"; printf '\n'; }
  exit 0
fi
[[ "${CONFIRM_TARGETED_RESTART_SMOKE:-}" == "TARGETED8_${TARGET8_EXPERIMENT}_DP32" ]] || {
  echo "set CONFIRM_TARGETED_RESTART_SMOKE=TARGETED8_${TARGET8_EXPERIMENT}_DP32" >&2; exit 2;
}
[[ ! -e "$TARGET8_SMOKE_ROOT" ]] || { echo "smoke root already exists" >&2; exit 2; }
mkdir -p "$TARGET8_SMOKE_ROOT/logs"
cleanup_failed_submit() { rmdir "$TARGET8_SMOKE_ROOT/logs" "$TARGET8_SMOKE_ROOT" 2>/dev/null || true; }
trap cleanup_failed_submit ERR
job=$("${command[@]}")
trap - ERR
printf '%s\n' "$job"
