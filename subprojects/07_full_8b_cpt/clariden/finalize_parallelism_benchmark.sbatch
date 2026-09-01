#!/usr/bin/env bash
#SBATCH --job-name=full8b-dp-bench-gate
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_BENCHMARK_ROOT:?set benchmark root}"
: "${FULL8_RECIPE:?set derived sanitized recipe}"
: "${FULL8_PROFILES:?set derived sanitized profiles}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/finalize_parallelism_benchmark.py" \
  --profiles "$FULL8_PROFILES" \
  --benchmark-contract "$FULL8_BENCHMARK_ROOT/benchmark_contract.json" \
  --control-root "$FULL8_BENCHMARK_ROOT/control_dp32" \
  --candidate-root "$FULL8_BENCHMARK_ROOT/candidate_dp64" \
  --control-repeat-root "$FULL8_BENCHMARK_ROOT/control_dp32_repeat" \
  --output "$FULL8_BENCHMARK_ROOT/promotion_receipt.json"

selected=$(python3 - "$FULL8_BENCHMARK_ROOT/promotion_receipt.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["selected_profile"])
PY
)
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_execution_profile.py" \
  --recipe "$FULL8_RECIPE" \
  --profiles "$FULL8_PROFILES" \
  --profile-id "$selected" \
  --output "$FULL8_BENCHMARK_ROOT/selected_execution_profile.json"
