#!/usr/bin/env bash
# Run one 13-panel shard inside a caller-owned Slurm step, preserving the GPUs
# Slurm actually assigned instead of assuming local device numbers.
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_VALIDATION_MANIFEST:?set validation manifest}"
: "${FULL8_HF_MODEL:?set HF model}"
: "${FULL8_HF_TOKENIZER:?set tokenizer}"
: "${FULL8_DOCVAL_OUTPUT:?set output root}"
: "${SLURM_ARRAY_TASK_ID:?set group 0 through 3}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign explicit GPUs}"
group=$SLURM_ARRAY_TASK_ID
[[ "$group" =~ ^[0-3]$ ]] || { echo "invalid document group" >&2; exit 2; }
mkdir -p "$FULL8_DOCVAL_OUTPUT"
/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
mapfile -t rows < <(/usr/bin/python3.11 - "$FULL8_VALIDATION_MANIFEST" "$group" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); group=int(sys.argv[2])
for i in range(group*4,min(group*4+4,len(d['panels']))):
 p=d['panels'][i]; print(f"{p['name']}\t{p['raw_jsonl']['path']}")
PY
)
IFS=',' read -r -a assigned <<<"$CUDA_VISIBLE_DEVICES"
[[ ${#rows[@]} -gt 0 && ${#assigned[@]} -ge ${#rows[@]} ]] || { echo "insufficient assigned GPUs" >&2; exit 2; }
pids=()
for rank in "${!rows[@]}"; do
  IFS=$'\t' read -r panel input <<<"${rows[$rank]}"
  CUDA_VISIBLE_DEVICES=${assigned[$rank]} uenv run pytorch/v2.9.1:v2 --view=default -- \
    python3 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/score_documents_hf.py" \
    --model "$FULL8_HF_MODEL" --tokenizer "$FULL8_HF_TOKENIZER" --input "$input" \
    --output "$FULL8_DOCVAL_OUTPUT/$panel.documents.jsonl" --receipt "$FULL8_DOCVAL_OUTPUT/$panel.receipt.json" \
    --device cuda --dtype bfloat16 --trust-remote-code \
    >"$FULL8_DOCVAL_OUTPUT/$panel.out" 2>"$FULL8_DOCVAL_OUTPUT/$panel.err" &
  pids+=("$!")
done
failure=0; for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
exit "$failure"
