#!/usr/bin/env bash
#SBATCH --job-name=full8b-supervisor-debug
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_STAGE_ROOT:?set stage root}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_INITIAL_MEGATRON:?set initial checkpoint}"
: "${FULL8_SELECTED_PROFILE:?set selected profile receipt}"
: "${FULL8_LAUNCH_GATE:?set launch gate receipt}"
: "${FULL8_PRELAUNCH_ROOT:?set completed prelaunch root}"
: "${FULL8_RECIPE:?set selected successor recipe}"
: "${FULL8_PROFILES:?set selected successor profiles}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set pinned Clariden leaf switch}"
[[ "$#" -eq 4 ]] || {
  echo "usage: $0 SEGMENT ATTEMPT ATTEMPT_START TRAIN_JOB" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "resource-aware supervisor must run on debug" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

prequeue_args=()
if [[ -n "${FULL8_PREQUEUED_MANIFEST:-}" ]]; then
  prequeue_args+=(--prequeued-manifest "$FULL8_PREQUEUED_MANIFEST")
fi
if [[ -n "${FULL8_PREQUEUE_SCHEDULE:-}" ]]; then
  prequeue_args+=(--prequeue-schedule "$FULL8_PREQUEUE_SCHEDULE")
fi

exec /usr/bin/python3.11 \
  "$FULL8_OPS_ROOT/scripts/supervise_campaign_resource_aware.py" \
  --code-root "$FULL8_CODE_ROOT" \
  --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --ops-root "$FULL8_OPS_ROOT" \
  --ops-bundle-receipt "$FULL8_OPS_BUNDLE_RECEIPT" \
  --stage-root "$FULL8_STAGE_ROOT" \
  --run-root "$FULL8_RUN_ROOT" \
  --initial-megatron "$FULL8_INITIAL_MEGATRON" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --launch-gate "$FULL8_LAUNCH_GATE" \
  --prelaunch-root "$FULL8_PRELAUNCH_ROOT" \
  --recipe "$FULL8_RECIPE" \
  --profiles "$FULL8_PROFILES" \
  --train-leaf-switch "$FULL8_TRAIN_LEAF_SWITCH" \
  "${prequeue_args[@]}" \
  --segment-id "$1" --attempt "$2" --attempt-start "$3" --train-job-id "$4"
