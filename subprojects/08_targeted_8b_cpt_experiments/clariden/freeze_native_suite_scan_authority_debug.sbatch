#!/usr/bin/env bash
#SBATCH --job-name=h2g-native-query
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
frozen_examples=${H2G_NATIVE_FROZEN_EXAMPLES:-/iopsstor/scratch/cscs/fffoivos/evals/full8_native_greek_3cp_20260812/frozen_examples_v2/examples.jsonl}
published_audit=${H2G_NATIVE_PUBLISHED_AUDIT:-/capstor/scratch/cscs/fffoivos/benchmark_contamination_audits/runs/20260812T171530Z-native-greek-v1/publish_payload/audit_receipt.json}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "native-suite query freeze requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
audit_scripts="$H2G_CODE_ROOT/frozen_native_contamination_audit"
queries="$H2G_STAGE_ROOT/data/native_suite_queries.jsonl"
summary="$H2G_STAGE_ROOT/data/native_suite_queries.summary.json"
receipt="$H2G_STAGE_ROOT/receipts/native_suite_scan_authority.json"
for path in "$queries" "$summary" "$receipt"; do
  [[ ! -e "$path" ]] || { echo "immutable native-suite query output exists: $path" >&2; exit 2; }
done
mkdir -p "$H2G_STAGE_ROOT/data" "$H2G_STAGE_ROOT/receipts"

uenv run pytorch/v2.6.0:v1 --view=default -- \
  python3 "$audit_scripts/build_decontamination_queries.py" \
    --frozen-examples-jsonl "$frozen_examples" \
    --benchmarks all \
    --output-jsonl "$queries" \
    --summary-json "$summary"
uenv run pytorch/v2.6.0:v1 --view=default -- \
  python3 "$subproject/scripts/freeze_native_suite_scan_authority.py" \
    --queries-jsonl "$queries" \
    --queries-summary "$summary" \
    --frozen-examples "$frozen_examples" \
    --builder-script "$audit_scripts/build_decontamination_queries.py" \
    --scanner-script "$audit_scripts/audit_benchmark_contamination_parquet.py" \
    --published-audit-receipt "$published_audit" \
    --output "$receipt"
