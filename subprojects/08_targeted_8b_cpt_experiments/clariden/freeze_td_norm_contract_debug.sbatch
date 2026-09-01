#!/usr/bin/env bash
#SBATCH --job-name=h2g-td-norms
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
export H2G_CODE_ROOT H2G_CODE_RECEIPT
input="$H2G_STAGE_ROOT/assets/init/8b_td_hf_raw"
historical="$H2G_CODE_ROOT/frozen_td_tools/historical_layer11_retok_td_manifest.json"
output="$H2G_STAGE_ROOT/receipts/td_row_norm_contract.json"
[[ -s "$H2G_STAGE_ROOT/receipts/8b_td_hf_materialization.json" && -d "$input" ]] || {
  echo "8B TD materialization is missing" >&2; exit 2;
}
[[ ! -e "$output" ]] || { echo "norm contract already exists: $output"; exit 0; }
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_td_row_norm_contract.py" \
  --verified-8b-td-root "$input" --historical-td-manifest "$historical" --output "$output"
