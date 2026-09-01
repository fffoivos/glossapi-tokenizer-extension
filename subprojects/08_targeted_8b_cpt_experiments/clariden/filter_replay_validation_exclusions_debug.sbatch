#!/usr/bin/env bash
#SBATCH --job-name=h2g-replay-valdrop
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_REPLAY_RAW_JSONL:?set rebuilt raw replay JSONL}"
: "${H2G_REPLAY_RAW_MANIFEST:?set rebuilt raw replay manifest}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "replay validation exclusion requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" \
  "$subproject/scripts/filter_replay_validation_exclusions.py" \
    --input-jsonl "$H2G_REPLAY_RAW_JSONL" \
    --input-manifest "$H2G_REPLAY_RAW_MANIFEST" \
    --validation-exclusions "$H2G_STAGE_ROOT/data/validation_text_exclusions.parquet" \
    --validation-receipt "$H2G_STAGE_ROOT/receipts/reused_validation_panels.json" \
    --output-jsonl "$H2G_STAGE_ROOT/data/replay_selected_validation_disjoint.jsonl" \
    --dropped-jsonl "$H2G_STAGE_ROOT/data/replay_validation_dropped.jsonl" \
    --output-manifest "$H2G_STAGE_ROOT/receipts/replay_validation_exclusion.json"
