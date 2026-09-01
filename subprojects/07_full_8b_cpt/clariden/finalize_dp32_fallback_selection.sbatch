#!/usr/bin/env bash
#SBATCH --job-name=full8b-dp32-select
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable selection code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable selection code receipt}"
: "${FULL8_PARITY_CODE_ROOT:?set immutable code root used by parity jobs}"
: "${FULL8_PARITY_CODE_BUNDLE_RECEIPT:?set immutable parity code receipt}"
: "${FULL8_STAGE_ROOT:?set completed sanitized D0 stage}"
: "${FULL8_PARITY_ROOT:?set completed synchronous DP32 parity root}"
: "${FULL8_BENCHMARK_ROOT:?set new DP32 selection evidence root}"
: "${FULL8_RECIPE:?set derived sanitized recipe}"
: "${FULL8_PROFILES:?set derived sanitized profiles}"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_PARITY_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_PARITY_CODE_ROOT" --receipt "$FULL8_PARITY_CODE_BUNDLE_RECEIPT" --kind scientific

[[ ! -e "$FULL8_BENCHMARK_ROOT" ]] || {
  echo "DP32 selection root already exists: $FULL8_BENCHMARK_ROOT" >&2
  exit 2
}
mkdir -p "$FULL8_BENCHMARK_ROOT"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/finalize_dp32_fallback_selection.py" \
  --code-root "$FULL8_CODE_ROOT" \
  --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --parity-code-root "$FULL8_PARITY_CODE_ROOT" \
  --parity-code-bundle-receipt "$FULL8_PARITY_CODE_BUNDLE_RECEIPT" \
  --recipe "$FULL8_RECIPE" \
  --profiles "$FULL8_PROFILES" \
  --stage-root "$FULL8_STAGE_ROOT" \
  --parity-root "$FULL8_PARITY_ROOT" \
  --output "$FULL8_BENCHMARK_ROOT/promotion_receipt.json"

uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_execution_profile.py" \
  --recipe "$FULL8_RECIPE" \
  --profiles "$FULL8_PROFILES" \
  --profile-id dp32_16node \
  --output "$FULL8_BENCHMARK_ROOT/selected_execution_profile.json"
