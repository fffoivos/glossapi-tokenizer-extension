#!/usr/bin/env bash
# Run one frozen per-document panel group inside an already allocated srun step.
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable scientific code receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_VALIDATION_MANIFEST:?set frozen 13-panel validation manifest}"
: "${FULL8_HF_MODEL:?set one HF model/checkpoint path}"
: "${FULL8_HF_TOKENIZER:?set tokenizer path}"
: "${FULL8_DOCVAL_OUTPUT:?set checkpoint-specific output directory}"
: "${SLURM_ARRAY_TASK_ID:?set per-document group 0 through 3}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign explicit GPUs to this step}"
GROUP=$SLURM_ARRAY_TASK_ID
[[ "$GROUP" =~ ^[0-3]$ ]] || { echo "invalid document group: $GROUP" >&2; exit 2; }
mkdir -p "$FULL8_DOCVAL_OUTPUT"

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

mapfile -t rows < <(/usr/bin/python3.11 - "$FULL8_VALIDATION_MANIFEST" "$GROUP" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); group=int(sys.argv[2])
for index in range(group*4,min(group*4+4,len(d["panels"]))):
    panel=d["panels"][index]
    print(f"{index}\t{panel['name']}\t{panel['raw_jsonl']['path']}")
PY
)
IFS=',' read -r -a assigned_devices <<<"$CUDA_VISIBLE_DEVICES"
[[ "${#rows[@]}" -gt 0 && "${#assigned_devices[@]}" -ge "${#rows[@]}" ]] || {
  echo "group $GROUP has ${#rows[@]} panels but only ${#assigned_devices[@]} assigned GPUs" >&2
  exit 2
}

pids=()
for local_rank in "${!rows[@]}"; do
  IFS=$'\t' read -r panel_index panel_name input <<<"${rows[$local_rank]}"
  output="$FULL8_DOCVAL_OUTPUT/$panel_name.documents.jsonl"
  receipt="$FULL8_DOCVAL_OUTPUT/$panel_name.receipt.json"
  CUDA_VISIBLE_DEVICES=${assigned_devices[$local_rank]} \
    uenv run pytorch/v2.9.1:v2 --view=default -- \
    python3 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/score_documents_hf.py" \
    --model "$FULL8_HF_MODEL" --tokenizer "$FULL8_HF_TOKENIZER" \
    --input "$input" --output "$output" --receipt "$receipt" \
    --device cuda --dtype bfloat16 --trust-remote-code \
    >"$FULL8_DOCVAL_OUTPUT/$panel_name.out" \
    2>"$FULL8_DOCVAL_OUTPUT/$panel_name.err" &
  pids+=("$!")
done
failure=0
for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
exit "$failure"
