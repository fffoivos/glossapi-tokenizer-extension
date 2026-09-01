#!/usr/bin/env bash
#SBATCH --job-name=full8b-eval-continue-debug
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_STAGE_ROOT:?set frozen stage root}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_RECIPE:?set derived recipe}"
: "${FULL8_PROFILES:?set derived execution profiles}"
: "${FULL8_SELECTED_PROFILE:?set selected profile}"
: "${FULL8_LAUNCH_GATE:?set launch gate}"
: "${FULL8_EVALUATION_ITERATIONS:?set colon-separated checkpoint updates}"
: "${FULL8_EVAL_INDEX:?set evaluation-chain index}"
: "${FULL8_EVAL_ATTEMPT:?set evaluation attempt}"
: "${FULL8_EVAL_JOB_ID:?set completed evaluation job}"
: "${FULL8_SOURCE_SEGMENT_ID:?set source segment}"
: "${FULL8_SOURCE_TRAIN_JOB_ID:?set source training job}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "evaluation continuation must run on debug" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

next_args=()
[[ -z "${FULL8_NEXT_SEGMENT_ID:-}" ]] || next_args+=(--next-segment "$FULL8_NEXT_SEGMENT_ID")
[[ -z "${FULL8_NEXT_SEGMENT_START:-}" ]] || next_args+=(--next-segment-start "$FULL8_NEXT_SEGMENT_START")
[[ -z "${FULL8_PREQUEUED_MANIFEST:-}" ]] || next_args+=(--prequeued-manifest "$FULL8_PREQUEUED_MANIFEST")
[[ -z "${FULL8_PREQUEUE_SCHEDULE:-}" ]] || next_args+=(--prequeue-schedule "$FULL8_PREQUEUE_SCHEDULE")
exec /usr/bin/python3.11 \
  "$FULL8_OPS_ROOT/evaluation/continue_checkpoint_evaluation.py" \
  --code-root "$FULL8_CODE_ROOT" \
  --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --ops-root "$FULL8_OPS_ROOT" \
  --ops-bundle-receipt "$FULL8_OPS_BUNDLE_RECEIPT" \
  --stage-root "$FULL8_STAGE_ROOT" \
  --run-root "$FULL8_RUN_ROOT" \
  --recipe "$FULL8_RECIPE" \
  --profiles "$FULL8_PROFILES" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --launch-gate "$FULL8_LAUNCH_GATE" \
  --prelaunch-root "$FULL8_PRELAUNCH_ROOT" \
  --initial-megatron "$FULL8_INITIAL_MEGATRON" \
  --train-leaf-switch "$FULL8_TRAIN_LEAF_SWITCH" \
  --iterations "$FULL8_EVALUATION_ITERATIONS" \
  --index "$FULL8_EVAL_INDEX" \
  --attempt "$FULL8_EVAL_ATTEMPT" \
  --evaluation-job "$FULL8_EVAL_JOB_ID" \
  --continue-attempt "${FULL8_CONTINUE_ATTEMPT:-0}" \
  --source-segment "$FULL8_SOURCE_SEGMENT_ID" \
  --source-train-job "$FULL8_SOURCE_TRAIN_JOB_ID" \
  --next-train-job "${FULL8_NEXT_TRAIN_JOB_ID:-}" \
  "${next_args[@]}"
