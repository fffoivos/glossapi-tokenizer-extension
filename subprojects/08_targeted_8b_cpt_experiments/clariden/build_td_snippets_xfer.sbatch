#!/usr/bin/env bash
#SBATCH --job-name=h2g-td-snippets
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_TD_HPLT_JSONL:?set benchmark-clean HPLT JSONL}"
: "${H2G_TD_OPENARCHIVES_JSONL:?set benchmark-clean OpenArchives JSONL}"
: "${H2G_TD_XFER_RUNTIME_ROOT:?set immutable xfer runtime root}"
[[ "${SLURM_JOB_PARTITION:-}" == xfer && "${SLURM_NNODES:-0}" == 1 ]] || exit 2
[[ -z "${SLURM_JOB_GPUS:-}" ]] || { echo "TD coverage scan must be CPU-only" >&2; exit 2; }

/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
requirements="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/td_xfer_runtime_requirements_v1.txt"
runtime_receipt="$H2G_TD_XFER_RUNTIME_ROOT/runtime_receipt.json"
PYTHONDONTWRITEBYTECODE=1 "$H2G_TD_XFER_RUNTIME_ROOT/bin/python" \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/verify_td_xfer_runtime.py" \
  --runtime-root "$H2G_TD_XFER_RUNTIME_ROOT" --receipt "$runtime_receipt" \
  --requirements "$requirements" --code-root "$H2G_CODE_ROOT" \
  --code-receipt "$H2G_CODE_RECEIPT"

root="$H2G_STAGE_ROOT/assets/init/td_inputs"
receipt="$H2G_STAGE_ROOT/receipts/td_training_inputs.json"
[[ ! -e "$receipt" ]] || { echo "TD input receipt already exists: $receipt"; exit 0; }
export TOKENIZERS_PARALLELISM=true H2G_CODE_ROOT H2G_CODE_RECEIPT
if [[ ! -e "$root" ]]; then
  work=$(mktemp -d "$H2G_STAGE_ROOT/assets/init/.td_inputs.XXXXXX")
  cleanup() { rm -rf -- "$work"; }
  trap cleanup EXIT
  "$H2G_TD_XFER_RUNTIME_ROOT/bin/python" \
    "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/td_coverage_prepass_batched.py" \
    --reference-script "$H2G_CODE_ROOT/frozen_td_tools/td_coverage_prepass.py" \
    --input-jsonl "$H2G_TD_HPLT_JSONL" "$H2G_TD_OPENARCHIVES_JSONL" \
    --base-tokenizer "$H2G_STAGE_ROOT/assets/init/1p5b_parent_hf" \
    --student-tokenizer "$H2G_STAGE_ROOT/assets/tokenizer_148480" \
    --output-dir "$work/coverage" --new-id-start 131072 --new-id-end 148480 \
    --target-extended-tokens 2000000000 --snippets-per-token 100 --seed 20260523 \
    --no-require-nfc \
    --encode-batch-documents 32 --encode-batch-characters 250000 --encode-workers 16 \
    --max-encode-batches-in-flight 32 --parity-documents 256
  mv "$work" "$root"
  trap - EXIT
fi
for path in "$root/coverage/td_coverage_prepass.jsonl" "$root/coverage/td_coverage_summary.json" "$root/coverage/td_snippet_index/snippets.jsonl"; do
  [[ -s "$path" ]] || { echo "incomplete frozen TD input: $path" >&2; exit 2; }
done
"$H2G_TD_XFER_RUNTIME_ROOT/bin/python" \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/freeze_td_training_inputs.py" \
  --historical-manifest "$H2G_CODE_ROOT/frozen_td_tools/historical_layer11_retok_td_manifest.json" \
  --coverage-jsonl "$root/coverage/td_coverage_prepass.jsonl" \
  --coverage-summary "$root/coverage/td_coverage_summary.json" \
  --snippets-jsonl "$root/coverage/td_snippet_index/snippets.jsonl" \
  --coverage-seed 20260523 \
  --dataset-authority "$H2G_STAGE_ROOT/receipts/pre_main_data_authorities/dataset_authority.json" \
  --hplt-stage-b-receipt "$H2G_STAGE_ROOT/receipts/hplt_stage_b.json" \
  --openarchives-stage-b-receipt "$H2G_STAGE_ROOT/receipts/openarchives_stage_b.json" \
  --output-token-ids "$root/selected_token_ids.txt" --output-receipt "$receipt"
