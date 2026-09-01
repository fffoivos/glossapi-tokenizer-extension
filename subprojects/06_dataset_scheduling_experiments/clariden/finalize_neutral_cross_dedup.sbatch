#!/usr/bin/env bash
#SBATCH --job-name=neutral_dedup
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=06:00:00
set -euo pipefail

: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${NEUTRAL_RUN_ROOT:?set neutral pipeline root}"
: "${AGENT1_RUN_ROOT:?set frozen Agent-1 v5 run root}"
: "${POOL_CORPUS_RECEIPT:?set frozen training pool receipt}"
: "${TOKENIZER_DIR:?set frozen Mini tokenizer overlay}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.6.0:v1}
DATATROVE_RUNTIME_PYTHON=${DATATROVE_RUNTIME_PYTHON:-/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/.agent1-v5-clariden-debug-20260715T111552Z-30c72e9.coord/runtime/venv/bin/python}
output=${NEUTRAL_DEDUP_OUTPUT:-$NEUTRAL_RUN_ROOT/final-cross-dedup}
[[ ! -e "$output" ]] || { echo "refusing to replace $output" >&2; exit 2; }
bucket_args=()
for bucket in $(seq 0 31); do
  bucket_name=$(printf '%03d' "$bucket")
  bucket_args+=(--bucket-receipt "$NEUTRAL_RUN_ROOT/matches/bucket_${bucket_name}.jsonl.gz.receipt.json")
done
uenv run "$UENV_IMAGE" --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  "$DATATROVE_RUNTIME_PYTHON" \
  "$SCIENTIFIC_BUNDLE/evaluation/cross_deduplicate_neutral_external.py" finalize \
  --candidate-signatures-receipt "$NEUTRAL_RUN_ROOT/candidate-signatures/candidate_signatures_receipt.json" \
  "${bucket_args[@]}" \
  --training-config "$AGENT1_RUN_ROOT/run_contract.json" \
  --training-signature-manifest "$AGENT1_RUN_ROOT/signature_manifest.json" \
  --training-combined-manifest "$AGENT1_RUN_ROOT/release-pre-dedup/manifests/combined_manifest.json" \
  --training-exact-root "$AGENT1_RUN_ROOT/60-dedup/exact-index" \
  --training-exact-receipts "$AGENT1_RUN_ROOT/60-dedup/exact-index/receipts" \
  --pool-corpus-receipt "$POOL_CORPUS_RECEIPT" --tokenizer-dir "$TOKENIZER_DIR" \
  --output-root "$output" --target-tokens 15000000 --hash-workers 32
