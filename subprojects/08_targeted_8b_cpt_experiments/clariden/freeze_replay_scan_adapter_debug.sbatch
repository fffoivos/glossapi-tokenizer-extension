#!/usr/bin/env bash
#SBATCH --job-name=h2g-replay-adapter
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_REPLAY_JSONL:?set rebuilt pre-decontamination replay JSONL}"
: "${H2G_REPLAY_MIX_MANIFEST:?set rebuilt replay mix manifest}"
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "replay adapter freeze requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
python_env=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
uenv run pytorch/v2.9.1:v2 --view=default -- "$python_env" \
  "$subproject/scripts/freeze_replay_scan_adapter.py" \
    --replay-jsonl "$H2G_REPLAY_JSONL" \
    --mix-manifest "$H2G_REPLAY_MIX_MANIFEST" \
    --output "$H2G_STAGE_ROOT/configs/replay_scan_adapter_v1.json"
