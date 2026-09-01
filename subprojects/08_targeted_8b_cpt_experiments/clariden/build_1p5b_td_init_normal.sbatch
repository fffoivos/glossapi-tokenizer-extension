#!/usr/bin/env bash
#SBATCH --job-name=h2g-td-1p5b
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=72
#SBATCH --mem=440G
#SBATCH --time=04:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
[[ "${SLURM_JOB_PARTITION:-}" == normal && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
exec /usr/bin/env bash \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/run_1p5b_td_init_common.sh"
