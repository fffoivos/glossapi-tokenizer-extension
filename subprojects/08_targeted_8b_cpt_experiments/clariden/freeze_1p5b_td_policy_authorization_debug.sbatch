#!/usr/bin/env bash
#SBATCH --job-name=h2g-td-policy-auth
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_TD_POLICY_APPROVAL_SOURCE:?set explicit owner approval source}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
policy="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/1p5b_td_acceptance_policy_v2.json"
output="$H2G_STAGE_ROOT/receipts/1p5b_td_policy_authorization.json"
[[ ! -e "$output" ]] || { echo "immutable TD policy authorization exists: $output" >&2; exit 2; }
python3 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_1p5b_td_policy_authorization.py" \
  --policy "$policy" --approval-source "$H2G_TD_POLICY_APPROVAL_SOURCE" --output "$output"
