#!/usr/bin/env bash
#SBATCH --job-name=h2g-phase3-catalog
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
: "${H2G_PHASE2_REALIZED:?set realized Phase-2 document ledger}"
: "${H2G_PHASE2_REALIZED_RECEIPT:?set realized Phase-2 document-ledger receipt}"
: "${H2G_OA_CATALOG:?set OpenArchives candidate catalog}"
: "${H2G_OA_CATALOG_RECEIPT:?set OpenArchives candidate-catalog receipt}"
: "${H2G_OA_SOURCE_JSONL:?set OpenArchives candidate source JSONL}"
: "${H2G_FOREIGN_CATALOG:?set foreign replay candidate catalog}"
: "${H2G_FOREIGN_CATALOG_RECEIPT:?set foreign replay candidate-catalog receipt}"
: "${H2G_FOREIGN_SOURCE_JSONL:?set foreign replay candidate source JSONL}"
: "${H2G_OLD_GREEK_CATALOG:?set Old-Greek replay candidate catalog}"
: "${H2G_OLD_GREEK_CATALOG_RECEIPT:?set Old-Greek candidate-catalog receipt}"
: "${H2G_OLD_GREEK_SOURCE_JSONL:?set Old-Greek candidate source JSONL}"
: "${H2G_PHASE3_DUPLICATE_EXCEPTIONS:?set frozen owner-authorized exact duplicate exclusions}"

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "Phase-3 catalog construction requires one debug node" >&2
  exit 2
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
  "$data_python" "$subproject/scripts/build_phase3_unseen_catalog.py" \
    --phase2-realized "$H2G_PHASE2_REALIZED" \
    --phase2-realized-receipt "$H2G_PHASE2_REALIZED_RECEIPT" \
    --openarchives-catalog "$H2G_OA_CATALOG" \
    --openarchives-catalog-receipt "$H2G_OA_CATALOG_RECEIPT" \
    --openarchives-source-jsonl "$H2G_OA_SOURCE_JSONL" \
    --openarchives-target-tokens 1577226077 \
    --foreign-replay-catalog "$H2G_FOREIGN_CATALOG" \
    --foreign-replay-catalog-receipt "$H2G_FOREIGN_CATALOG_RECEIPT" \
    --foreign-replay-source-jsonl "$H2G_FOREIGN_SOURCE_JSONL" \
    --foreign-replay-target-tokens 399297741 \
    --old-greek-replay-catalog "$H2G_OLD_GREEK_CATALOG" \
    --old-greek-replay-catalog-receipt "$H2G_OLD_GREEK_CATALOG_RECEIPT" \
    --old-greek-replay-source-jsonl "$H2G_OLD_GREEK_SOURCE_JSONL" \
    --old-greek-replay-target-tokens 19964888 \
    --duplicate-exceptions "$H2G_PHASE3_DUPLICATE_EXCEPTIONS" \
    --output-dir "$H2G_STAGE_ROOT/data/phase3_unseen" \
    --output-receipt "$H2G_STAGE_ROOT/receipts/phase3_unseen_catalog.json"
