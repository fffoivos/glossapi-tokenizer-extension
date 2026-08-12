#!/usr/bin/env bash
#SBATCH --job-name=benchmark-contam
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=01:25:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue

set -euo pipefail
: "${AUDIT_CODE_ROOT:?set immutable code root}"
: "${AUDIT_CODE_RECEIPT:?set immutable code receipt}"
: "${AUDIT_RUN_ROOT:?set audit run root}"
: "${AUDIT_DATASET_RUN_ROOT:?set exact HF publication run root}"
: "${AUDIT_FROZEN_EXAMPLES:?set frozen benchmark examples JSONL}"
: "${AUDIT_PYTHON:?set Python with pyarrow}"

verify="$AUDIT_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation/verify_eval_bundle.py"
builder="$AUDIT_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/build_decontamination_queries.py"
scanner="$AUDIT_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/audit_benchmark_contamination_parquet.py"
finalizer="$AUDIT_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/finalize_benchmark_contamination_audit.py"
for path in "$verify" "$builder" "$scanner" "$finalizer" "$AUDIT_FROZEN_EXAMPLES"; do
  [[ -f "$path" ]] || { echo "missing required input: $path" >&2; exit 2; }
done
"$AUDIT_PYTHON" "$verify" --root "$AUDIT_CODE_ROOT" --receipt "$AUDIT_CODE_RECEIPT"

mkdir -p "$AUDIT_RUN_ROOT/control" "$AUDIT_RUN_ROOT/logs"
stop_file="$AUDIT_RUN_ROOT/control/scan_stop_requested"
rm -f "$stop_file"
requeue_requested=0
request_requeue() { requeue_requested=1; : > "$stop_file"; }
request_stop() { : > "$stop_file"; }
trap request_requeue USR1
trap request_stop TERM INT

queries="$AUDIT_RUN_ROOT/queries.jsonl"
query_summary="$AUDIT_RUN_ROOT/queries_summary.json"
if [[ ! -e "$queries" && ! -e "$query_summary" ]]; then
  "$AUDIT_PYTHON" "$builder" \
    --frozen-examples-jsonl "$AUDIT_FROZEN_EXAMPLES" \
    --benchmarks all --output-jsonl "$queries" --summary-json "$query_summary"
fi
[[ -f "$queries" && -f "$query_summary" ]]

dataset_release="$AUDIT_DATASET_RUN_ROOT/release"
manifest="$dataset_release/manifests/anonymization_manifest.json"
publication="$AUDIT_DATASET_RUN_ROOT/publication/receipt.json"
[[ -d "$dataset_release/data" && -f "$manifest" && -f "$publication" ]]

children=()
for lane in 0 1 2 3; do
  (
    "$AUDIT_PYTHON" "$scanner" \
      --queries-jsonl "$queries" \
      --corpus-root "$dataset_release" \
      --corpus-manifest "$manifest" \
      --publication-receipt "$publication" \
      --output-root "$AUDIT_RUN_ROOT/scan" \
      --lane "$lane" --lanes 4 \
      --workers "${AUDIT_WORKERS_PER_LANE:-64}" \
      --batch-size "${AUDIT_BATCH_SIZE:-1024}" \
      --stop-file "$stop_file"
  ) >"$AUDIT_RUN_ROOT/logs/lane_${lane}.out" 2>"$AUDIT_RUN_ROOT/logs/lane_${lane}.err" &
  children+=("$!")
done
status=0
for child in "${children[@]}"; do
  wait "$child" || status=1
done
if ((requeue_requested)); then
  rm -f "$stop_file"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
if [[ -e "$stop_file" ]]; then
  exit 1
fi
((status == 0)) || { echo "one or more scan lanes failed" >&2; exit 1; }

shard_receipts=$(find "$AUDIT_RUN_ROOT/scan/shards" -type f -name '*.receipt.json' | wc -l)
[[ "$shard_receipts" -eq 431 ]] || {
  echo "scan ended without all 431 shard receipts (got $shard_receipts)" >&2
  exit 1
}
if [[ ! -e "$AUDIT_RUN_ROOT/publish_payload" ]]; then
  "$AUDIT_PYTHON" "$finalizer" \
    --queries-jsonl "$queries" --queries-summary "$query_summary" \
    --corpus-manifest "$manifest" --publication-receipt "$publication" \
    --scan-root "$AUDIT_RUN_ROOT/scan" \
    --output-dir "$AUDIT_RUN_ROOT/publish_payload"
fi
[[ -f "$AUDIT_RUN_ROOT/publish_payload/publish_manifest.json" ]]
