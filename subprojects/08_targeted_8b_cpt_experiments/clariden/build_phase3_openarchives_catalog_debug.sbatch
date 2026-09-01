#!/usr/bin/env bash
#SBATCH --job-name=h2g-phase3-oa-catalog
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
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "Phase-3 OpenArchives catalog requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
uenv run pytorch/v2.9.1:v2 --view=default -- env PYTHONDONTWRITEBYTECODE=1 \
  "$data_python" "$subproject/scripts/verify_data_runtime.py" \
    --runtime-root "$data_runtime_root" --receipt "$data_runtime_root/runtime_receipt.json" \
    --requirements "$subproject/configs/data_runtime_requirements_v1.txt" \
    --code-root "$H2G_CODE_ROOT" --code-receipt "$H2G_CODE_RECEIPT" >/dev/null
uenv run pytorch/v2.9.1:v2 --view=default -- env TOKENIZERS_PARALLELISM=true \
  "$data_python" "$subproject/scripts/build_jsonl_document_catalog.py" \
    --stream phase3_openarchives_candidates --pool openarchives \
    --input-jsonl "$H2G_STAGE_ROOT/data/phase3_openarchives_candidates_stage_b.jsonl" \
    --input-receipt "$H2G_STAGE_ROOT/receipts/phase3_openarchives_candidates_stage_b.json" \
    --tokenizer-root "$H2G_STAGE_ROOT/assets/tokenizer_148480" \
    --tokenizer-receipt "$H2G_STAGE_ROOT/receipts/tokenizer_148480.json" \
    --batch-size 512 \
    --output-jsonl "$H2G_STAGE_ROOT/catalogs/phase3_openarchives_candidates_documents.jsonl" \
    --output-receipt "$H2G_STAGE_ROOT/receipts/phase3_openarchives_candidates_catalog.json"
