#!/usr/bin/env bash
#SBATCH --job-name=h2g-native-excl
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code-bundle receipt}"
: "${H2G_STAGE_ROOT:?set immutable experiment stage root}"
: "${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}"
: "${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}"
RELEASE_ROOT=/capstor/scratch/cscs/fffoivos/hf_v2_anonymized_releases/20260811T121833Z/run
MATCH_TABLE=/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/publish_payload/qa_document_line_matches.parquet

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "native-suite exclusion build requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

manifest="$H2G_STAGE_ROOT/data/native_suite_training_exclusions.parquet"
receipt="$H2G_STAGE_ROOT/receipts/native_suite_training_exclusions.json"
[[ ! -e "$manifest" && ! -e "$receipt" ]] || { echo "immutable native-suite exclusion output exists" >&2; exit 2; }
mkdir -p "$H2G_STAGE_ROOT/data" "$H2G_STAGE_ROOT/receipts"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$H2G_DATA_PYTHON" "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_native_suite_training_exclusions.py" \
    --match-table "$MATCH_TABLE" \
    --release-root "$RELEASE_ROOT" \
    --output-manifest "$manifest" \
    --output-receipt "$receipt"
