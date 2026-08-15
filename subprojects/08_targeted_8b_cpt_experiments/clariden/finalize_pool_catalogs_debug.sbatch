#!/usr/bin/env bash
#SBATCH --job-name=t8b-final-catalogs
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
PARENT_POOL=${TARGET8_PARENT_POOL_RECEIPT:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260807T063000Z-d0-v3/inventory/pool_corpus_receipt.json}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "catalog finalization must run on debug" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
stage="$TARGET8_STAGE_ROOT/experiment_a"
mkdir -p "$stage/inventory"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/finalize_targeted_pool_catalogs.py" \
  --stage-root "$stage" --input-receipt "$stage/binary_contract/input_receipt.json" \
  --heldout-manifest "$stage/binary_contract/heldout_manifest.json" \
  --academic-pool-receipt "$stage/receipts/academic_pool_receipt.json" \
  --poly-pool-receipt "$stage/receipts/poly_pool_receipt.json" --parent-pool-receipt "$PARENT_POOL" \
  --schedule-seed 20260811 --output "$stage/inventory/pool_corpus_receipt.json"
