#!/usr/bin/env bash
#SBATCH --job-name=full8b-launch-gate
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
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable bundle receipt}"
: "${FULL8_STAGE_ROOT:?set frozen data root}"
: "${FULL8_PRELAUNCH_ROOT:?set completed prelaunch root}"
: "${FULL8_BENCHMARK_ROOT:?set completed benchmark root}"
: "${FULL8_INITIAL_MEGATRON:?set verified TP2 initialization root}"
: "${FULL8_INITIAL_CHECKPOINT_RECEIPT:?set checkpoint-tree receipt}"
: "${FULL8_INITIAL_HF_RECEIPT:?set corrected iteration-zero HF anchor receipt}"
: "${FULL8_INITIAL_VALIDATION_RECEIPT:?set corrected initial validation receipt}"
: "${FULL8_INITIAL_GREEKMMLU_RECEIPT:?set corrected iteration-zero GreekMMLU receipt}"
: "${FULL8_INITIAL_PER_DOCUMENT_ROOT:?set corrected iteration-zero per-document root}"
: "${FULL8_PACKED_PAYLOAD_INTEGRITY:?set completed packed-payload hash receipt}"
: "${FULL8_CONVERSION_SMOKE_RECEIPT:?set completed checkpoint-conversion smoke receipt}"
: "${FULL8_GRACEFUL_STOP_SMOKE:?set completed graceful-stop/restart smoke receipt}"
: "${FULL8_RUN_ROOT:?set new production run root}"
: "${FULL8_LAUNCH_AUTHORIZATION:?set explicit production authorization}"
[[ "$FULL8_LAUNCH_AUTHORIZATION" == APERTUS8B_FULL_MIXED_CPT ]] || {
  echo "invalid production launch authorization" >&2
  exit 2
}

recipe="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json"
profiles="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/execution_profiles.json"
owner="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/owner_decisions_20260805.json"
selected="$FULL8_BENCHMARK_ROOT/selected_execution_profile.json"
environment="$FULL8_PRELAUNCH_ROOT/launch_environment.json"
gate="$FULL8_PRELAUNCH_ROOT/launch_gate.json"
conversion="$FULL8_CONVERSION_SMOKE_RECEIPT"
[[ ! -e "$environment" && ! -e "$gate" && ! -e "$FULL8_RUN_ROOT" ]] || {
  echo "refusing to overwrite launch evidence or production run root" >&2
  exit 2
}

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/capture_launch_environment.py" \
  --selected-profile "$selected" \
  --storage-path /capstor/scratch/cscs/fffoivos/runs \
  --output "$environment"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/build_launch_gate.py" \
  --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --code-root "$FULL8_CODE_ROOT" \
  --recipe "$recipe" \
  --profiles "$profiles" \
  --selected-profile "$selected" \
  --owner-decisions "$owner" \
  --hf-visibility "$FULL8_PRELAUNCH_ROOT/hf_visibility.json" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --packed-receipt "$FULL8_STAGE_ROOT/inventory/packed_corpus_receipt.json" \
  --packed-payload-integrity "$FULL8_PACKED_PAYLOAD_INTEGRITY" \
  --schedule-manifest "$FULL8_STAGE_ROOT/schedules/schedule_manifest.json" \
  --validation-manifest "$FULL8_STAGE_ROOT/validation/validation_manifest.json" \
  --token-bytes-receipt "$FULL8_STAGE_ROOT/validation/token_utf8_byte_lengths.receipt.json" \
  --benchmark-receipt "$FULL8_BENCHMARK_ROOT/promotion_receipt.json" \
  --graceful-stop-smoke "$FULL8_GRACEFUL_STOP_SMOKE" \
  --initial-validation "$FULL8_INITIAL_VALIDATION_RECEIPT" \
  --initial-greekmmlu "$FULL8_INITIAL_GREEKMMLU_RECEIPT" \
  --initial-per-document-root "$FULL8_INITIAL_PER_DOCUMENT_ROOT" \
  --initial-megatron "$FULL8_INITIAL_MEGATRON" \
  --initial-checkpoint-receipt "$FULL8_INITIAL_CHECKPOINT_RECEIPT" \
  --initial-hf-receipt "$FULL8_INITIAL_HF_RECEIPT" \
  --conversion-smoke "$conversion" \
  --launch-environment "$environment" \
  --output "$gate"

common=(
  FULL8_CODE_ROOT="$FULL8_CODE_ROOT"
  FULL8_CODE_BUNDLE_RECEIPT="$FULL8_CODE_BUNDLE_RECEIPT"
  FULL8_STAGE_ROOT="$FULL8_STAGE_ROOT"
  FULL8_RUN_ROOT="$FULL8_RUN_ROOT"
  FULL8_INITIAL_MEGATRON="$FULL8_INITIAL_MEGATRON"
  FULL8_PRELAUNCH_ROOT="$FULL8_PRELAUNCH_ROOT"
  FULL8_SELECTED_PROFILE="$selected"
  FULL8_LAUNCH_GATE="$gate"
)
env "${common[@]}" DRY_RUN=1 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/submit_production.sh" \
  >"$FULL8_PRELAUNCH_ROOT/production_submission_dry_run.out" \
  2>"$FULL8_PRELAUNCH_ROOT/production_submission_dry_run.err"

env "${common[@]}" DRY_RUN=0 CONFIRM_GPU_LAUNCH=APERTUS8B_FULL_MIXED_CPT \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/submit_production.sh" \
  | tee "$FULL8_PRELAUNCH_ROOT/production_submission.json"
