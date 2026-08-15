#!/usr/bin/env bash
#SBATCH --job-name=h2g-artifacts
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_MANIFEST_MODE \
  H2G_PRODUCER_COMPATIBILITY H2G_SCALE H2G_PROFILE_PROMOTION \
  H2G_TRAINING_RUN_PERMIT H2G_ARTIFACT_MANIFEST_OUTPUT H2G_MEGATRON_RECEIPT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "$H2G_SCALE" == 8b || "$H2G_SCALE" == 1p5b ]] || { echo "H2G_SCALE must be 8b or 1p5b" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "artifact-manifest freeze requires one debug node" >&2; exit 2; }
for path in "$H2G_PROFILE_PROMOTION" "$H2G_TRAINING_RUN_PERMIT" "$H2G_ARTIFACT_MANIFEST_OUTPUT"; do
  case "$path" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "artifact-manifest authorities must stay inside the stage receipts root: $path" >&2; exit 2 ;; esac
done
case "$H2G_MEGATRON_RECEIPT" in
  "$H2G_STAGE_ROOT"/receipts/*) ;;
  *) echo "Megatron receipt must stay inside the stage receipts root" >&2; exit 2 ;;
esac
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
receipts="$H2G_STAGE_ROOT/receipts"
sentinel="$H2G_STAGE_ROOT/evaluation/greekmmlu_sentinels/sentinel_manifest.json"
declare -a rows=(
  "historical_asset_inventory=$receipts/historical_asset_inventory.json"
  "benchmark_union_and_exclusions=$receipts/pre_main_data_authorities/benchmark_union_authority.json"
  "dataset_rebuild_and_lineage=$receipts/pre_main_data_authorities/dataset_authority.json"
  "heldout_overlap_audit=$receipts/pre_main_data_authorities/heldout_overlap_authority.json"
  "replay_split=$receipts/replay_split.json"
  "weighted_blend_and_index_caches=$receipts/pre_main_data_authorities/blend_cache_authority.json"
  "training_megatron_runtime=$H2G_MEGATRON_RECEIPT"
  "tokenizer_148480=$receipts/tokenizer_148480.json"
  "offline_validation_panels=$receipts/reused_validation_panels.json"
  "historical_online_validation_binaries=$receipts/historical_online_validation_binaries.json"
  "sentinel_freeze=$sentinel"
  "evaluation_code_bundle=$receipts/evaluation_code_bundle.json"
  "legacy_public_evaluator_contract=$receipts/legacy_public_evaluator_contract.json"
  "statistical_decision_contract=$receipts/statistical_decision_contract.json"
)
if [[ "$H2G_SCALE" == 8b ]]; then
  rows+=(
    "8b_init_roundtrip=$receipts/8b_init_roundtrip.json"
    "8b_restart_parity=$H2G_PROFILE_PROMOTION"
    "8b_training_run_permit=$H2G_TRAINING_RUN_PERMIT"
  )
else
  : "${H2G_LR_SELECTION:?set the exact 1.5B LR-selection receipt}"
  case "$H2G_LR_SELECTION" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "LR-selection receipt must stay inside the stage receipts root" >&2; exit 2 ;; esac
  rows+=(
    "1p5b_td_init=$receipts/1p5b_td_initialization_v2.json"
    "1p5b_init_roundtrip=$receipts/1p5b_init_roundtrip.json"
    "1p5b_profile_candidate=$H2G_PROFILE_PROMOTION"
    "1p5b_lr_selection=$H2G_LR_SELECTION"
    "1p5b_training_run_permit=$H2G_TRAINING_RUN_PERMIT"
  )
fi
case "$H2G_MANIFEST_MODE" in
  pre_timing)
    partial=(--allow-partial)
    ;;
  pre_authorization)
    : "${H2G_PRODUCTION_TIMING:?set exact production-timing receipt}"
    : "${H2G_ALLOCATION_SCHEDULE:?set exact allocation-schedule receipt}"
    rows+=(
      "production_timing=$H2G_PRODUCTION_TIMING"
      "allocation_schedule=$H2G_ALLOCATION_SCHEDULE"
    )
    partial=(--allow-partial)
    ;;
  final)
    : "${H2G_PRODUCTION_TIMING:?set exact production-timing receipt}"
    : "${H2G_ALLOCATION_SCHEDULE:?set exact allocation-schedule receipt}"
    : "${H2G_OWNER_AUTHORIZATION:?set exact owner-authorization receipt}"
    rows+=(
      "production_timing=$H2G_PRODUCTION_TIMING"
      "allocation_schedule=$H2G_ALLOCATION_SCHEDULE"
      "owner_production_authorization=$H2G_OWNER_AUTHORIZATION"
    )
    partial=()
    ;;
  *) echo "invalid H2G_MANIFEST_MODE" >&2; exit 2 ;;
esac
output="$H2G_ARTIFACT_MANIFEST_OUTPUT"
for path in "${H2G_PRODUCTION_TIMING:-}" "${H2G_ALLOCATION_SCHEDULE:-}" "${H2G_OWNER_AUTHORIZATION:-}"; do
  [[ -z "$path" ]] && continue
  case "$path" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "terminal authority must stay inside the stage receipts root: $path" >&2; exit 2 ;; esac
done
args=()
for row in "${rows[@]}"; do args+=(--artifact "$row"); done
export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_artifact_manifest.py" \
  --gate-stage pre_main --scale "$H2G_SCALE" --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  "${partial[@]}" "${args[@]}" --output "$output"
