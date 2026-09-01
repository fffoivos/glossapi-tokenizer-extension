#!/usr/bin/env bash
#SBATCH --job-name=h2g-ext-artifacts
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_GATE_STAGE \
  H2G_MANIFEST_MODE H2G_PRODUCER_COMPATIBILITY; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ -z "${H2G_SCALE:-}" ]] || {
  echo "extension manifests are joint and forbid H2G_SCALE" >&2; exit 2;
}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "extension artifact freeze requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

receipts="$H2G_STAGE_ROOT/receipts"
partial=()
case "$H2G_GATE_STAGE:$H2G_MANIFEST_MODE" in
  pre_extension:pre_authorization)
    for name in H2G_8B_UPDATE_3218_PERMIT H2G_1P5B_UPDATE_3218_PERMIT; do
      [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
    done
    rows=(
      "main_8b_update_3218_checkpoint_permit=$H2G_8B_UPDATE_3218_PERMIT"
      "main_1p5b_update_3218_checkpoint_permit=$H2G_1P5B_UPDATE_3218_PERMIT"
      "cross_scale_realized_sample_ledger_match_through_3218=$receipts/cross_scale_realized_ledger_match.json"
      "same_stack_sentinel_calibration_state=$receipts/greekmmlu_sentinel_calibration_authority.json"
      "phase_3_unseen_blend_and_capacity_receipt=$receipts/phase3_authority.json"
      "phase_local_cursor_guard_receipt=$receipts/phase_local_cursor_guard.json"
      "constant_floor_scheduler_resume_receipt=$receipts/constant_floor_scheduler_resume.json"
    )
    output="$receipts/artifact_manifest_pre_extension_authorization.json"
    partial=(--allow-partial)
    ;;
  pre_extension:final)
    for name in H2G_8B_UPDATE_3218_PERMIT H2G_1P5B_UPDATE_3218_PERMIT; do
      [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
    done
    rows=(
      "main_8b_update_3218_checkpoint_permit=$H2G_8B_UPDATE_3218_PERMIT"
      "main_1p5b_update_3218_checkpoint_permit=$H2G_1P5B_UPDATE_3218_PERMIT"
      "cross_scale_realized_sample_ledger_match_through_3218=$receipts/cross_scale_realized_ledger_match.json"
      "same_stack_sentinel_calibration_state=$receipts/greekmmlu_sentinel_calibration_authority.json"
      "phase_3_unseen_blend_and_capacity_receipt=$receipts/phase3_authority.json"
      "phase_local_cursor_guard_receipt=$receipts/phase_local_cursor_guard.json"
      "constant_floor_scheduler_resume_receipt=$receipts/constant_floor_scheduler_resume.json"
      "owner_extension_authorization=$receipts/owner_extension_authorization.json"
    )
    output="$receipts/artifact_manifest_pre_extension.json"
    ;;
  pre_second_extension:final)
    rows=(
      "both_update_3456_checkpoint_permits=$receipts/checkpoint_pair_3456.json"
      "phase_3_3456_to_3457_resume_receipts=$receipts/resume_pair_3456.json"
    )
    output="$receipts/artifact_manifest_pre_second_extension.json"
    ;;
  pre_finalization:final)
    final="$H2G_STAGE_ROOT/evaluation/final_matched_evidence"
    rows=(
      "all_required_source_panel_receipts=$final/source_panel_authority.json"
      "all_required_greekmmlu_receipts=$final/greekmmlu_authority.json"
      "all_required_native_suite_receipts=$final/native_suite_authority.json"
      "selection_authorization_receipt=$final/selection_authorization.json"
    )
    output="$receipts/artifact_manifest_pre_finalization.json"
    ;;
  *)
    echo "invalid H2G_GATE_STAGE:H2G_MANIFEST_MODE" >&2
    exit 2
    ;;
esac

args=()
for row in "${rows[@]}"; do args+=(--artifact "$row"); done
export H2G_CODE_ROOT H2G_CODE_RECEIPT PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_artifact_manifest.py" \
  --gate-stage "$H2G_GATE_STAGE" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  "${partial[@]}" "${args[@]}" --output "$output"
