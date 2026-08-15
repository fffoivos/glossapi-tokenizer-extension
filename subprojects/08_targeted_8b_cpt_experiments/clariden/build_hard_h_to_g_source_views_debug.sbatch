#!/usr/bin/env bash
#SBATCH --job-name=h2g-source-views
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
release_root=${H2G_RELEASE_ROOT:-/capstor/scratch/cscs/fffoivos/hf_v2_anonymized_releases/20260811T121833Z/run}
greek_replay=${H2G_GREEK_REPLAY_PARQUET:-/iopsstor/scratch/cscs/fffoivos/cpt_corpus/greek_replay/greek_replay.parquet}
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
release_inspection=${H2G_RELEASE_INSPECTION_RECEIPT:-$H2G_STAGE_ROOT/receipts/hf_release_inspection.json}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "source-view build requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
native_manifest="$H2G_STAGE_ROOT/data/native_suite_training_exclusions.parquet"
native_receipt="$H2G_STAGE_ROOT/receipts/native_suite_training_exclusions.json"
replay_receipt="$H2G_STAGE_ROOT/receipts/replay_parquet_inspection.json"
validation_manifest="$H2G_STAGE_ROOT/data/validation_text_exclusions.parquet"
validation_receipt="$H2G_STAGE_ROOT/receipts/reused_validation_panels.json"
for path in "$release_inspection" "$native_manifest" "$native_receipt" "$replay_receipt" "$validation_manifest" "$validation_receipt"; do
  [[ -s "$path" ]] || { echo "required source-view input missing: $path" >&2; exit 2; }
done
mkdir -p "$H2G_STAGE_ROOT/data" "$H2G_STAGE_ROOT/receipts"
uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" \
  "$subproject/scripts/build_hard_h_to_g_source_views.py" \
    --release-root "$release_root" \
    --release-inspection-receipt "$release_inspection" \
    --native-exclusions "$native_manifest" \
    --native-exclusion-receipt "$native_receipt" \
    --greek-replay-parquet "$greek_replay" \
    --replay-inspection-receipt "$replay_receipt" \
    --validation-exclusions "$validation_manifest" \
    --validation-exclusion-receipt "$validation_receipt" \
    --output-root "$H2G_STAGE_ROOT/data/source_views" \
    --output-receipt "$H2G_STAGE_ROOT/receipts/source_views.json" \
    --workers 16
