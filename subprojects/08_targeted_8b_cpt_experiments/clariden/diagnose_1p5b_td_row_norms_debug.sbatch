#!/usr/bin/env bash
#SBATCH --job-name=h2g-td-norm-diag
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
output="$H2G_STAGE_ROOT/receipts/1p5b_td_row_norm_failure_diagnostic_${SLURM_JOB_ID}.json"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/diagnose_1p5b_td_row_norms.py" \
  --stage-root "$H2G_STAGE_ROOT" --output "$output"
