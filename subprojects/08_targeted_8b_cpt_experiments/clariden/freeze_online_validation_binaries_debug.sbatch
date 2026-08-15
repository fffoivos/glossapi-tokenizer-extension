#!/usr/bin/env bash
#SBATCH --job-name=h2g-online-val
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_VAL_DATA_DIR:?set exact historical online-validation binary root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "online-validation freeze requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_online_validation_binaries.py" \
  --root "$H2G_VAL_DATA_DIR" \
  --output "$H2G_STAGE_ROOT/receipts/historical_online_validation_binaries.json"
