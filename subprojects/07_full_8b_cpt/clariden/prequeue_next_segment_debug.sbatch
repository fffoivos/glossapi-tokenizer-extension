#!/usr/bin/env bash
#SBATCH --job-name=full8b-prequeue-controller
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "prequeue controller must run on debug" >&2
  exit 2
}
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_STAGE_ROOT:?set stage root}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_INITIAL_MEGATRON:?set initial checkpoint}"
: "${FULL8_SELECTED_PROFILE:?set selected profile}"
: "${FULL8_LAUNCH_GATE:?set launch gate}"
: "${FULL8_PRELAUNCH_ROOT:?set prelaunch root}"
: "${FULL8_RECIPE:?set recipe}"
: "${FULL8_PROFILES:?set execution profiles}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set train leaf switch}"
: "${FULL8_PREQUEUED_MANIFEST:?set prequeued manifest}"
: "${FULL8_PREQUEUE_SCHEDULE:?set prequeue schedule}"
: "${FULL8_PREQUEUE_SOURCE_SEGMENT:?set source segment}"
: "${FULL8_PREQUEUE_SOURCE_TRAIN_JOB:?set source train job}"
: "${FULL8_PREQUEUE_TARGET_SEGMENT:?set target segment}"
: "${FULL8_MIN_TRAIN_SECONDS:?set minimum train seconds}"
: "${FULL8_MAX_HOLD_SECONDS:?set maximum hold seconds}"

optional_args=()
if [[ -n "${FULL8_ELIGIBLE_AFTER_MINUTES:-}" ]]; then
  optional_args+=(--eligible-after-minutes "$FULL8_ELIGIBLE_AFTER_MINUTES")
fi
if [[ "${FULL8_PREQUEUE_TEST_ONLY:-0}" == 1 ]]; then
  optional_args+=(--test-only)
fi

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

receipt="$FULL8_RUN_ROOT/orchestration/prequeue_controller_receipts/segment_${FULL8_PREQUEUE_TARGET_SEGMENT}.json"
exec /usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/prequeue_next_segment.py" \
  --scientific-root "$FULL8_CODE_ROOT" \
  --scientific-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --ops-root "$FULL8_OPS_ROOT" \
  --ops-receipt "$FULL8_OPS_BUNDLE_RECEIPT" \
  --stage-root "$FULL8_STAGE_ROOT" \
  --run-root "$FULL8_RUN_ROOT" \
  --initial-megatron "$FULL8_INITIAL_MEGATRON" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --launch-gate "$FULL8_LAUNCH_GATE" \
  --prelaunch-root "$FULL8_PRELAUNCH_ROOT" \
  --recipe "$FULL8_RECIPE" \
  --profiles "$FULL8_PROFILES" \
  --train-leaf-switch "$FULL8_TRAIN_LEAF_SWITCH" \
  --manifest "$FULL8_PREQUEUED_MANIFEST" \
  --schedule "$FULL8_PREQUEUE_SCHEDULE" \
  --source-segment "$FULL8_PREQUEUE_SOURCE_SEGMENT" \
  --source-train-job "$FULL8_PREQUEUE_SOURCE_TRAIN_JOB" \
  --target-segment "$FULL8_PREQUEUE_TARGET_SEGMENT" \
  --minimum-train-seconds "$FULL8_MIN_TRAIN_SECONDS" \
  --maximum-hold-seconds "$FULL8_MAX_HOLD_SECONDS" \
  "${optional_args[@]}" \
  --output "$receipt"
