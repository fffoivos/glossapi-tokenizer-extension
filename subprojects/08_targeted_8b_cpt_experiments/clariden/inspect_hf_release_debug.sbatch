#!/usr/bin/env bash
#SBATCH --job-name=t8b-inspect-hf
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_RELEASE_ROOT:?set anonymized release root}"
: "${H2G_STAGE_ROOT:?set targeted preparation root}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}

[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "release inspection must run on debug" >&2
  exit 2
}
[[ "${SLURM_NNODES:-0}" == 1 ]] || { echo "release inspection requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"
output="$H2G_STAGE_ROOT/receipts/hf_release_inspection.json"
mkdir -p "$(dirname "$output")" "$H2G_STAGE_ROOT/logs"
[[ ! -e "$output" ]] || { echo "immutable inspection receipt already exists" >&2; exit 2; }

uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/inspect_hf_release.py" \
    --release-root "$H2G_RELEASE_ROOT" \
    --output "$output"
