#!/usr/bin/env bash
#SBATCH --job-name=h2g-campaign-v3
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_SCALE \
  APERTUS_CANONICAL_ROOT APERTUS_CANONICAL_CODE_RECEIPT \
  H2G_INITIALIZATION_PERMIT H2G_MEGATRON_RECEIPT H2G_QUALIFICATION_CONTRACT \
  H2G_PRODUCER_COMPATIBILITY H2G_TRAINING_RUN_PERMIT \
  H2G_PROFILE_PROMOTION H2G_PRODUCTION_TIMING H2G_ALLOCATION_SCHEDULE \
  H2G_PRE_MAIN_LAUNCH_GATE H2G_PHASE1_CACHE_RECEIPT H2G_PHASE2_CACHE_RECEIPT EVAL_VENV \
  EVAL_CODE_ROOT EVAL_CODE_RECEIPT EVAL_SOURCE_CONTRACT \
  EVAL_SOURCE_MANIFEST EVAL_SOURCE_GATE EVAL_EXCLUSIONS; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "$H2G_SCALE" == 8b || "$H2G_SCALE" == 1p5b ]] || {
  echo "H2G_SCALE must be 8b or 1p5b" >&2; exit 2;
}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "canonical contract compilation requires one debug node" >&2; exit 2;
}
for path in "$H2G_TRAINING_RUN_PERMIT" "$H2G_PROFILE_PROMOTION" \
  "$H2G_PRODUCTION_TIMING" "$H2G_ALLOCATION_SCHEDULE" "$H2G_PRE_MAIN_LAUNCH_GATE"; do
  case "$path" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "canonical campaign authorities must stay inside the stage receipts root: $path" >&2; exit 2 ;; esac
done

verify_code="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify_code" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
/usr/bin/python3.11 "$verify_code" \
  --root "$APERTUS_CANONICAL_ROOT" --receipt "$APERTUS_CANONICAL_CODE_RECEIPT" --kind efficiency

canonical_data_root=${H2G_CANONICAL_DATA_ROOT:-$H2G_STAGE_ROOT/canonical/pre_main_data_v2}
phase3_spec=${H2G_PHASE3_DATA_PATH_SPEC:-$H2G_STAGE_ROOT/data/phases/phase3/phase_data_path.json}
phase3_receipt=${H2G_PHASE3_CACHE_RECEIPT:-$H2G_STAGE_ROOT/receipts/phase_3_blend_cache.json}
phase3_root=${H2G_PHASE3_CACHE_ROOT:-$H2G_STAGE_ROOT/data/phases/phase3/cache}

export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_canonical_campaign_contracts.py" \
  --scale "$H2G_SCALE" \
  --stage-root "$H2G_STAGE_ROOT" \
  --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" \
  --canonical-runner-root "$APERTUS_CANONICAL_ROOT" \
  --canonical-data-root "$canonical_data_root" \
  --initialization-permit "$H2G_INITIALIZATION_PERMIT" \
  --megatron-receipt "$H2G_MEGATRON_RECEIPT" \
  --qualification-contract "$H2G_QUALIFICATION_CONTRACT" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --training-run-permit "$H2G_TRAINING_RUN_PERMIT" \
  --profile-promotion "$H2G_PROFILE_PROMOTION" \
  --production-timing "$H2G_PRODUCTION_TIMING" \
  --allocation-schedule "$H2G_ALLOCATION_SCHEDULE" \
  --pre-main-launch-gate "$H2G_PRE_MAIN_LAUNCH_GATE" \
  --phase1-cache-receipt "$H2G_PHASE1_CACHE_RECEIPT" \
  --phase2-cache-receipt "$H2G_PHASE2_CACHE_RECEIPT" \
  --phase3-data-path-spec "$phase3_spec" \
  --phase3-cache-receipt "$phase3_receipt" --phase3-cache-root "$phase3_root" \
  --eval-venv "$EVAL_VENV" \
  --native-eval-code-root "$EVAL_CODE_ROOT" \
  --native-eval-code-receipt "$EVAL_CODE_RECEIPT" \
  --native-source-contract "$EVAL_SOURCE_CONTRACT" \
  --native-source-manifest "$EVAL_SOURCE_MANIFEST" \
  --native-source-gate "$EVAL_SOURCE_GATE" \
  --native-exclusions "$EVAL_EXCLUSIONS"
