#!/usr/bin/env bash
#SBATCH --job-name=h2g-overlap-schema
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code-bundle receipt}"
: "${H2G_STAGE_ROOT:?set immutable experiment stage root}"
: "${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}"
: "${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}"
MATCH_TABLE=/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/publish_payload/qa_document_line_matches.parquet

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "overlap-schema inspection requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

output="$H2G_STAGE_ROOT/receipts/native_overlap_schema.json"
[[ ! -e "$output" ]] || { echo "immutable receipt exists: $output" >&2; exit 2; }
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$H2G_DATA_PYTHON" "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/inspect_native_overlap_schema.py" \
    --match-table "$MATCH_TABLE" --output "$output"
