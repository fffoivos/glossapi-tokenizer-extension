#!/usr/bin/env bash
#SBATCH --job-name=t8b-b-gate-launch
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${TARGET8_STAGE_ROOT:?set B targeted preparation root}"
: "${TARGET8_RESTART_SMOKE_ROOT:?set completed restart-smoke root}"
: "${TARGET8_RESTART_SMOKE_JOB_ID:?set completed restart-smoke job id}"
: "${TARGET8_INITIAL_ANCHOR_ROOT:?set completed initial anchor root}"
: "${TARGET8_INITIAL_CHECKPOINT_RECEIPT:?set parent checkpoint receipt}"
: "${TARGET8_PARENT_CHECKPOINT_ROOT:?set parent checkpoint root}"
: "${TARGET8_GRACEFUL_STOP_SMOKE:?set passing graceful-stop receipt}"
: "${TARGET8_NESTED_SBATCH_PROOF:?set bundle-bound nested proof}"
: "${TARGET8_FINAL_PRELAUNCH_ROOT:?set new final prelaunch root}"
: "${TARGET8_PRODUCTION_RUN_ROOT:?set new production run root}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "B gate/launch controller must run on debug" >&2; exit 2; }
[[ ! -e "$TARGET8_FINAL_PRELAUNCH_ROOT" ]] || { echo "immutable final prelaunch root exists" >&2; exit 2; }
[[ ! -e "$TARGET8_PRODUCTION_RUN_ROOT" ]] || { echo "production run root exists" >&2; exit 2; }

verify="$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify" --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 "$verify" --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

stage="$TARGET8_STAGE_ROOT/experiment_b"
selected="$stage/contracts/selected_execution_profile.json"
recipe="$stage/contracts/recipe_8b_full_mixed.sanitized.json"
profiles="$stage/contracts/execution_profiles.sanitized.json"
restart="$TARGET8_RESTART_SMOKE_ROOT/restart_smoke_receipt.json"
for path in "$selected" "$recipe" "$profiles" "$restart" \
  "$TARGET8_INITIAL_ANCHOR_ROOT/initial_validation_receipt.json" \
  "$TARGET8_INITIAL_ANCHOR_ROOT/greekmmlu_receipt.json" \
  "$TARGET8_INITIAL_ANCHOR_ROOT/conversion_greekmmlu_smoke_receipt.json" \
  "$TARGET8_NESTED_SBATCH_PROOF" "$TARGET8_INITIAL_CHECKPOINT_RECEIPT" \
  "$TARGET8_GRACEFUL_STOP_SMOKE"; do
  [[ -s "$path" ]] || { echo "required B gate input missing: $path" >&2; exit 2; }
done

mkdir -p "$TARGET8_FINAL_PRELAUNCH_ROOT"
ln -s "$TARGET8_RESTART_SMOKE_ROOT" "$TARGET8_FINAL_PRELAUNCH_ROOT/restart_smoke"
ln -s "$TARGET8_INITIAL_ANCHOR_ROOT" "$TARGET8_FINAL_PRELAUNCH_ROOT/initial_anchor"

leaf=$(/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/derive_completed_job_leaf.py" \
  --job-id "$TARGET8_RESTART_SMOKE_JOB_ID" --restart-smoke "$restart" \
  --output "$TARGET8_FINAL_PRELAUNCH_ROOT/completed_smoke_leaf.json")
exclude=$("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$leaf" 16)
[[ -n "$exclude" ]] || { echo "empty production leaf exclusion" >&2; exit 2; }

/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/capture_launch_environment.py" \
  --selected-profile "$selected" --leaf-switch "$leaf" --leaf-exclusion "$exclude" \
  --storage-path /capstor/scratch/cscs/fffoivos/runs \
  --output "$TARGET8_FINAL_PRELAUNCH_ROOT/launch_environment.json"

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_targeted_launch_gate.py" \
  --experiment B --code-root "$FULL8_CODE_ROOT" --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --frozen-contract "$stage/contracts/frozen_experiment_contract.json" \
  --training-assets "$stage/contracts/training_assets_receipt.json" \
  --recipe "$recipe" --profiles "$profiles" --selected-profile "$selected" \
  --owner-authorization "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/owner_authorization.json" \
  --pool-receipt "$stage/inventory/pool_corpus_receipt.json" \
  --packed-receipt "$stage/packed_corpus_receipt.json" \
  --packed-integrity "$stage/packed_payload_integrity_receipt.json" \
  --schedule-manifest "$stage/schedules/schedule_manifest.json" \
  --validation-manifest "$stage/validation/validation_manifest.json" \
  --restart-smoke "$restart" --graceful-stop-smoke "$TARGET8_GRACEFUL_STOP_SMOKE" \
  --initial-validation "$TARGET8_INITIAL_ANCHOR_ROOT/initial_validation_receipt.json" \
  --initial-greekmmlu "$TARGET8_INITIAL_ANCHOR_ROOT/greekmmlu_receipt.json" \
  --initial-per-document-root "$TARGET8_INITIAL_ANCHOR_ROOT/per_document" \
  --initial-checkpoint-receipt "$TARGET8_INITIAL_CHECKPOINT_RECEIPT" \
  --conversion-smoke "$TARGET8_INITIAL_ANCHOR_ROOT/conversion_greekmmlu_smoke_receipt.json" \
  --launch-environment "$TARGET8_FINAL_PRELAUNCH_ROOT/launch_environment.json" \
  --nested-sbatch-proof "$TARGET8_NESTED_SBATCH_PROOF" \
  --output "$TARGET8_FINAL_PRELAUNCH_ROOT/launch_gate.json"

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_targeted_operational_gate.py" \
  --experiment B --scientific-root "$FULL8_CODE_ROOT" --scientific-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --operational-root "$FULL8_OPS_ROOT" --operational-receipt "$FULL8_OPS_BUNDLE_RECEIPT" \
  --launch-gate "$TARGET8_FINAL_PRELAUNCH_ROOT/launch_gate.json" --selected-profile "$selected" \
  --launch-environment "$TARGET8_FINAL_PRELAUNCH_ROOT/launch_environment.json" \
  --nested-sbatch-proof "$TARGET8_NESTED_SBATCH_PROOF" \
  --output "$TARGET8_FINAL_PRELAUNCH_ROOT/operational_launch_gate.json"

env \
  TARGET8_EXPERIMENT=B \
  FULL8_CODE_ROOT="$FULL8_CODE_ROOT" FULL8_CODE_BUNDLE_RECEIPT="$FULL8_CODE_BUNDLE_RECEIPT" \
  FULL8_OPS_ROOT="$FULL8_OPS_ROOT" FULL8_OPS_BUNDLE_RECEIPT="$FULL8_OPS_BUNDLE_RECEIPT" \
  FULL8_STAGE_ROOT="$stage" FULL8_RUN_ROOT="$TARGET8_PRODUCTION_RUN_ROOT" \
  FULL8_INITIAL_MEGATRON="$TARGET8_PARENT_CHECKPOINT_ROOT" FULL8_PRELAUNCH_ROOT="$TARGET8_FINAL_PRELAUNCH_ROOT" \
  FULL8_SELECTED_PROFILE="$selected" FULL8_LAUNCH_GATE="$TARGET8_FINAL_PRELAUNCH_ROOT/launch_gate.json" \
  FULL8_OPERATIONAL_LAUNCH_GATE="$TARGET8_FINAL_PRELAUNCH_ROOT/operational_launch_gate.json" \
  FULL8_TRAIN_LEAF_SWITCH="$leaf" TARGET8_PARENT_CHECKPOINT_ROOT="$TARGET8_PARENT_CHECKPOINT_ROOT" \
  DRY_RUN=0 CONFIRM_GPU_LAUNCH=TARGETED8_CPT_B \
  bash "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/submit_targeted_production.sh" \
  | tee "$TARGET8_FINAL_PRELAUNCH_ROOT/production_submission.txt"
