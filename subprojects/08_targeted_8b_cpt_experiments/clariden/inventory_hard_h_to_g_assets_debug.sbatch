#!/usr/bin/env bash
#SBATCH --job-name=h2g-assets
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_STAGE_ROOT:?set immutable experiment stage root}"
: "${H2G_CODE_RECEIPT:?set immutable code-bundle receipt}"

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "asset inventory must run on debug" >&2
  exit 2
}
[[ "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "asset inventory requires exactly one debug node" >&2
  exit 2
}

verify="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
output="$H2G_STAGE_ROOT/receipts/historical_asset_inventory.json"
mkdir -p "$H2G_STAGE_ROOT/receipts" "$H2G_STAGE_ROOT/logs"
[[ ! -e "$output" ]] || {
  echo "immutable asset inventory already exists: $output" >&2
  exit 2
}

/usr/bin/python3.11 "$subproject/scripts/inventory_hard_h_to_g_assets.py" \
  --asset-spec "$subproject/configs/hard_h_to_g_assets_v1.json" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --output "$output"
