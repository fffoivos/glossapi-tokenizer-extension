#!/usr/bin/env bash
#SBATCH --job-name=t8b-launch-gate
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_EXPERIMENT:?set A or B}"
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_STAGE_ROOT:?set frozen targeted experiment root}"
: "${FULL8_PRELAUNCH_ROOT:?set targeted prelaunch root}"
: "${FULL8_INITIAL_CHECKPOINT_RECEIPT:?set initial checkpoint receipt}"
: "${FULL8_GRACEFUL_STOP_SMOKE:?set passing graceful-stop receipt}"
: "${FULL8_NESTED_SBATCH_PROOF:?set nested-submit proof}"
: "${FULL8_LAUNCH_GATE:?set new immutable launch-gate path}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "launch gate must run on debug" >&2; exit 2; }
[[ "$TARGET8_EXPERIMENT" == A || "$TARGET8_EXPERIMENT" == B ]] || { echo "TARGET8_EXPERIMENT must be A or B" >&2; exit 2; }

name=$(printf '%s' "$TARGET8_EXPERIMENT" | tr '[:upper:]' '[:lower:]')
stage="$FULL8_STAGE_ROOT/experiment_$name"
args=()
if [[ "$TARGET8_EXPERIMENT" == A ]]; then
  args+=(
    --decontamination-summary "$stage/receipts/decontamination_summary.json"
    --selected-pools "$stage/receipts/selected_pools_receipt.json"
  )
fi
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_targeted_launch_gate.py" \
  --experiment "$TARGET8_EXPERIMENT" --code-root "$FULL8_CODE_ROOT" \
  --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --frozen-contract "$stage/contracts/frozen_experiment_contract.json" \
  --training-assets "$stage/contracts/training_assets_receipt.json" \
  --recipe "$stage/contracts/recipe_8b_full_mixed.sanitized.json" \
  --profiles "$stage/contracts/execution_profiles.sanitized.json" \
  --selected-profile "$stage/contracts/selected_execution_profile.json" \
  --owner-authorization "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/owner_authorization.json" \
  --pool-receipt "$stage/inventory/pool_corpus_receipt.json" \
  --packed-receipt "$stage/packed_corpus_receipt.json" \
  --packed-integrity "$stage/packed_payload_integrity_receipt.json" \
  --schedule-manifest "$stage/schedules/schedule_manifest.json" \
  --validation-manifest "$stage/validation/validation_manifest.json" \
  "${args[@]}" \
  --restart-smoke "$FULL8_PRELAUNCH_ROOT/restart_smoke/restart_smoke_receipt.json" \
  --graceful-stop-smoke "$FULL8_GRACEFUL_STOP_SMOKE" \
  --initial-validation "$FULL8_PRELAUNCH_ROOT/initial_anchor/initial_validation_receipt.json" \
  --initial-greekmmlu "$FULL8_PRELAUNCH_ROOT/initial_anchor/greekmmlu_receipt.json" \
  --initial-per-document-root "$FULL8_PRELAUNCH_ROOT/initial_anchor/per_document" \
  --initial-checkpoint-receipt "$FULL8_INITIAL_CHECKPOINT_RECEIPT" \
  --conversion-smoke "$FULL8_PRELAUNCH_ROOT/initial_anchor/conversion_greekmmlu_smoke_receipt.json" \
  --launch-environment "$FULL8_PRELAUNCH_ROOT/launch_environment.json" \
  --nested-sbatch-proof "$FULL8_NESTED_SBATCH_PROOF" \
  --output "$FULL8_LAUNCH_GATE"
