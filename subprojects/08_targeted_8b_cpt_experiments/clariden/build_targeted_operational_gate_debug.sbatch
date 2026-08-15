#!/usr/bin/env bash
#SBATCH --job-name=t8b-ops-gate
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_EXPERIMENT:?set A or B}"
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_LAUNCH_GATE:?set passing targeted launch gate}"
: "${FULL8_SELECTED_PROFILE:?set frozen selected profile}"
: "${FULL8_LAUNCH_ENVIRONMENT:?set fresh scheduler snapshot}"
: "${FULL8_NESTED_SBATCH_PROOF:?set nested-submit proof}"
: "${FULL8_OPERATIONAL_LAUNCH_GATE:?set new operational gate path}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "operational gate must run on debug" >&2; exit 2; }

args=()
if [[ "$TARGET8_EXPERIMENT" == A ]]; then
  : "${FULL8_PREQUEUE_SCHEDULE:?A requires frozen prequeue schedule}"
  args+=(--prequeue-schedule "$FULL8_PREQUEUE_SCHEDULE")
fi
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_targeted_operational_gate.py" \
  --experiment "$TARGET8_EXPERIMENT" \
  --scientific-root "$FULL8_CODE_ROOT" --scientific-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --operational-root "$FULL8_OPS_ROOT" --operational-receipt "$FULL8_OPS_BUNDLE_RECEIPT" \
  --launch-gate "$FULL8_LAUNCH_GATE" --selected-profile "$FULL8_SELECTED_PROFILE" \
  --launch-environment "$FULL8_LAUNCH_ENVIRONMENT" --nested-sbatch-proof "$FULL8_NESTED_SBATCH_PROOF" \
  "${args[@]}" --output "$FULL8_OPERATIONAL_LAUNCH_GATE"
