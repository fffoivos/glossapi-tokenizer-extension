#!/usr/bin/env bash
#SBATCH --job-name=h2g-native-scan
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
scan_label=${H2G_NATIVE_SCAN_LABEL:-pre_filter}
case "$scan_label" in pre_filter|post_filter) ;; *) echo "unsupported scan label: $scan_label" >&2; exit 2 ;; esac

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "native-suite replay scan requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
scanner="$H2G_CODE_ROOT/frozen_native_contamination_audit/audit_benchmark_contamination_parquet.py"
queries="$H2G_STAGE_ROOT/data/native_suite_queries.jsonl"
query_authority="$H2G_STAGE_ROOT/receipts/native_suite_scan_authority.json"
if [[ "$scan_label" == pre_filter ]]; then
  corpus_root="$H2G_STAGE_ROOT/data/replay_native_scan_corpus"
  corpus_receipt="$H2G_STAGE_ROOT/receipts/replay_native_scan_corpus.json"
else
  corpus_root="$H2G_STAGE_ROOT/data/replay_native_postfilter_scan_corpus"
  corpus_receipt="$H2G_STAGE_ROOT/receipts/replay_native_postfilter_scan_corpus.json"
fi
manifest="$corpus_root/manifest.json"
publication="$corpus_root/publication_receipt.json"
scan_root="$H2G_STAGE_ROOT/data/native_suite_replay_scan_${scan_label}"
exclusions="$H2G_STAGE_ROOT/data/native_suite_replay_exclusions_${scan_label}.parquet"
receipt="$H2G_STAGE_ROOT/receipts/native_suite_replay_scan_${scan_label}.json"
for path in "$queries" "$query_authority" "$corpus_receipt" "$manifest" "$publication"; do
  [[ -s "$path" ]] || { echo "native-suite scan prerequisite missing: $path" >&2; exit 2; }
done
for path in "$scan_root" "$exclusions" "$receipt"; do
  [[ ! -e "$path" ]] || { echo "immutable native-suite scan output exists: $path" >&2; exit 2; }
done

uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" "$scanner" \
    --queries-jsonl "$queries" \
    --corpus-root "$corpus_root" \
    --corpus-manifest "$manifest" \
    --publication-receipt "$publication" \
    --output-root "$scan_root" \
    --k 8 \
    --minimum-short-question-tokens 3 \
    --max-gap-tokens 50 \
    --max-gap-tokens-short 5 \
    --workers "${SLURM_CPUS_PER_TASK:-64}" \
    --batch-size 2048 \
    --lane 0 --lanes 1
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" "$subproject/scripts/finalize_native_training_exclusions.py" \
    --queries-jsonl "$queries" \
    --query-authority "$query_authority" \
    --corpus-manifest "$manifest" \
    --publication-receipt "$publication" \
    --scan-root "$scan_root" \
    --output-parquet "$exclusions" \
    --output-receipt "$receipt"
if [[ "$scan_label" == post_filter ]]; then
  H2G_RECEIPT="$receipt" uenv run pytorch/v2.9.1:v2 --view=default -- "$data_python" - <<'PY'
import json, os
from pathlib import Path
r=json.loads(Path(os.environ["H2G_RECEIPT"]).read_text())
if r["counts"]["excluded_documents"] != 0 or r["counts"]["strong_match_rows"] != 0:
    raise SystemExit("post-filter native-suite scan was not clean")
print({"post_filter_zero_strong_matches": True})
PY
fi
