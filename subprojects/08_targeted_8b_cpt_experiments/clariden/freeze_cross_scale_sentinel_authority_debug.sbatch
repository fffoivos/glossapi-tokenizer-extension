#!/usr/bin/env bash
#SBATCH --job-name=h2g-sentinel-authority
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_8B_CALIBRATION_RESULT:?set exact canonical 8B calibration result}"
: "${H2G_1P5B_CALIBRATION_RESULT:?set exact canonical 1.5B calibration result}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "cross-scale sentinel authority requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
root="$H2G_STAGE_ROOT/evaluation/greekmmlu_sentinels"
output="$H2G_STAGE_ROOT/receipts/greekmmlu_sentinel_calibration_authority.json"
[[ ! -e "$output" ]] || { echo "immutable authority exists: $output" >&2; exit 2; }
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 \
  "$subproject/scripts/freeze_cross_scale_sentinel_authority.py" \
  --manifest "$root/sentinel_manifest.json" \
  --8b-calibration "$H2G_8B_CALIBRATION_RESULT" \
  --1p5b-calibration "$H2G_1P5B_CALIBRATION_RESULT" \
  --output "$output"
