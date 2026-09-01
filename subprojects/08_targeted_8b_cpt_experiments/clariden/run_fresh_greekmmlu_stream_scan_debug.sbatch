#!/usr/bin/env bash
#SBATCH --job-name=h2g-gmmlu-scan
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_STREAM_NAME:?set one frozen stream name}"
: "${H2G_CORPUS_JSONL:?set rebuilt stream JSONL}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "fresh GreekMMLU scan requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
queries="$H2G_STAGE_ROOT/data/greekmmlu_public_queries.jsonl"
query_receipt="$H2G_STAGE_ROOT/receipts/greekmmlu_query_regeneration.json"
scan_root="$H2G_STAGE_ROOT/data/greekmmlu_scans/$H2G_STREAM_NAME"
receipt="$H2G_STAGE_ROOT/receipts/greekmmlu_scan_${H2G_STREAM_NAME}.json"
extra=()
if [[ "${H2G_GREEKMMLU_AUDIT_ONLY:-0}" == 1 ]]; then extra+=(--audit-only); fi

uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" "$subproject/scripts/run_fresh_greekmmlu_stream_scan.py" \
    --stream-name "$H2G_STREAM_NAME" \
    --corpus-jsonl "$H2G_CORPUS_JSONL" \
    --queries-jsonl "$queries" \
    --query-receipt "$query_receipt" \
    --scanner-script "$H2G_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/decontaminate.py" \
    --output-root "$scan_root" \
    --workers "${SLURM_CPUS_PER_TASK:-64}" \
    --output-receipt "$receipt" \
    "${extra[@]}"
