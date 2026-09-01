#!/usr/bin/env bash
#SBATCH --job-name=full8b-nested-submit-proof
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_NESTED_SMOKE_ROOT:?set nested-submit smoke root}"
mkdir -p "$FULL8_NESTED_SMOKE_ROOT/logs"
exec uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/submit_nested_sbatch_probe.py" \
  --code-root "$FULL8_CODE_ROOT" --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --output-root "$FULL8_NESTED_SMOKE_ROOT"
