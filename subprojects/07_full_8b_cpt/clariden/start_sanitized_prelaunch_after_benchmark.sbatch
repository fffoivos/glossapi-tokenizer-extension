#!/usr/bin/env bash
#SBATCH --job-name=full8b-prelaunch-coordinator
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set completed sanitized D0 stage}"
: "${FULL8_PRELAUNCH_ROOT:?set new prelaunch root}"
: "${FULL8_PRODUCTION_RUN_ROOT:?set new production run root}"
: "${FULL8_INITIAL_MEGATRON:?set canonical TP2 initialization root}"
: "${FULL8_BENCHMARK_ROOT:?set benchmark root gated by this job dependency}"
: "${FULL8_HF_VISIBILITY_SOURCE:?set completed public-dataset receipt}"
: "${FULL8_LAUNCH_AUTHORIZATION:?set explicit launch authorization}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set pinned Clariden leaf switch}"
: "${FULL8_COORDINATOR_ROOT:?set coordinator evidence root}"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
[[ -f "$FULL8_BENCHMARK_ROOT/promotion_receipt.json" && -f "$FULL8_BENCHMARK_ROOT/selected_execution_profile.json" ]] || {
  echo "benchmark promotion artifacts are absent" >&2; exit 2;
}
mkdir -p "$FULL8_COORDINATOR_ROOT"
launcher="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/submit_sanitized_prelaunch_and_launch.sh"
DRY_RUN=1 "$launcher" \
  >"$FULL8_COORDINATOR_ROOT/prelaunch_dry_run.out" \
  2>"$FULL8_COORDINATOR_ROOT/prelaunch_dry_run.err"
DRY_RUN=0 "$launcher" | tee "$FULL8_COORDINATOR_ROOT/prelaunch_submission.json"
