#!/usr/bin/env bash
#SBATCH --job-name=h2g-phase3-oa-clean
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
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "Phase-3 OpenArchives preparation requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/verify_data_runtime.py" \
    --runtime-root "$data_runtime_root" --receipt "$data_runtime_root/runtime_receipt.json" \
    --requirements "$subproject/configs/data_runtime_requirements_v1.txt" \
    --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" >/dev/null
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/prepare_greek_source_stream.py" \
    --pool openarchives \
    --selected-mix-receipt "$H2G_STAGE_ROOT/receipts/phase3_openarchives_candidates_raw.json" \
    --queries-jsonl "$H2G_STAGE_ROOT/data/greekmmlu_public_queries.jsonl" \
    --query-receipt "$H2G_STAGE_ROOT/receipts/greekmmlu_query_regeneration.json" \
    --e001-script "$H2G_CODE_ROOT/frozen_historical_dataset_tools/hplt_clean.py" \
    --scanner-script "$H2G_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/decontaminate.py" \
    --clean-jsonl "$H2G_STAGE_ROOT/data/phase3_openarchives_candidates_clean.jsonl" \
    --dropped-jsonl "$H2G_STAGE_ROOT/data/phase3_openarchives_candidates_greekmmlu_dropped.jsonl" \
    --output-receipt "$H2G_STAGE_ROOT/receipts/phase3_openarchives_candidates_clean.json" \
    --workers "${SLURM_CPUS_PER_TASK:-64}" --chunksize 256
