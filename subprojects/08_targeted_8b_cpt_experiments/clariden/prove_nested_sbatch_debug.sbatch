#!/usr/bin/env bash
# Prove that a controller inside the production uenv can submit a debug child.
#SBATCH --job-name=t8b-nested-submit-proof
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_NESTED_SMOKE_ROOT:?set new nested-submit smoke root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "nested-submit parent must run on debug" >&2
  exit 2
}
[[ ! -e "$FULL8_NESTED_SMOKE_ROOT" ]] || {
  echo "immutable nested-submit root already exists: $FULL8_NESTED_SMOKE_ROOT" >&2
  exit 2
}
mkdir -p "$FULL8_NESTED_SMOKE_ROOT/logs"
exec uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/submit_nested_sbatch_probe_debug.py" \
  --code-root "$FULL8_CODE_ROOT" --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --output-root "$FULL8_NESTED_SMOKE_ROOT"
