#!/usr/bin/env bash
#SBATCH --job-name=h2g-greek-prepare
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=01:28:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_POOL:?set hplt or openarchives}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
case "$H2G_POOL" in hplt|openarchives) ;; *) echo "invalid Greek source pool: $H2G_POOL" >&2; exit 2 ;; esac
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "Greek source preparation requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" "$subproject/scripts/prepare_greek_source_stream.py" \
    --pool "$H2G_POOL" \
    --selected-mix-receipt "$H2G_STAGE_ROOT/receipts/${H2G_POOL}_selected_raw.manifest.json" \
    --queries-jsonl "$H2G_STAGE_ROOT/data/greekmmlu_public_queries.jsonl" \
    --query-receipt "$H2G_STAGE_ROOT/receipts/greekmmlu_query_regeneration.json" \
    --e001-script "$H2G_CODE_ROOT/frozen_historical_dataset_tools/hplt_clean.py" \
    --scanner-script "$H2G_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/decontaminate.py" \
    --clean-jsonl "$H2G_STAGE_ROOT/data/${H2G_POOL}_prepared_clean.jsonl" \
    --dropped-jsonl "$H2G_STAGE_ROOT/data/${H2G_POOL}_greekmmlu_dropped.jsonl" \
    --output-receipt "$H2G_STAGE_ROOT/receipts/${H2G_POOL}_prepared_stream.json" \
    --workers "${SLURM_CPUS_PER_TASK:-64}" --chunksize 256
