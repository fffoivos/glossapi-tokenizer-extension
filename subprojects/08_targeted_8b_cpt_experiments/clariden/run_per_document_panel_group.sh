#!/usr/bin/env bash
set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable scientific code root}"
: "${H2G_CODE_RECEIPT:?set immutable scientific code receipt}"
: "${H2G_VALIDATION_MANIFEST:?set frozen 13-panel manifest}"
: "${H2G_HF_MODEL:?set verified HF checkpoint export}"
: "${H2G_HF_TOKENIZER:?set frozen tokenizer}"
: "${H2G_DOCVAL_OUTPUT:?set checkpoint-specific output root}"
: "${H2G_PANEL_GROUP:?set panel group 0 through 3}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must assign four explicit GPUs}"

group=$H2G_PANEL_GROUP
[[ "$group" =~ ^[0-3]$ ]] || { echo "invalid panel group: $group" >&2; exit 2; }
mkdir -p "$H2G_DOCVAL_OUTPUT"
/usr/bin/python3.11 \
  "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

mapfile -t rows < <(/usr/bin/python3.11 - "$H2G_VALIDATION_MANIFEST" "$group" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
group = int(sys.argv[2])
for index in range(group * 4, min(group * 4 + 4, len(manifest["panels"]))):
    panel = manifest["panels"][index]
    print(f"{panel['name']}\t{panel['raw_jsonl']['path']}")
PY
)
IFS=',' read -r -a devices <<<"$CUDA_VISIBLE_DEVICES"
[[ "${#rows[@]}" -gt 0 && "${#devices[@]}" -ge "${#rows[@]}" ]] || {
  echo "panel group $group has ${#rows[@]} panels but ${#devices[@]} GPUs" >&2
  exit 2
}

pids=()
for local_rank in "${!rows[@]}"; do
  IFS=$'\t' read -r panel input <<<"${rows[$local_rank]}"
  CUDA_VISIBLE_DEVICES=${devices[$local_rank]} \
    uenv run pytorch/v2.9.1:v2 --view=default -- \
    python3 "$H2G_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/score_documents_hf.py" \
      --model "$H2G_HF_MODEL" --tokenizer "$H2G_HF_TOKENIZER" \
      --input "$input" \
      --output "$H2G_DOCVAL_OUTPUT/$panel.documents.jsonl" \
      --receipt "$H2G_DOCVAL_OUTPUT/$panel.receipt.json" \
      --device cuda --dtype bfloat16 --trust-remote-code \
      >"$H2G_DOCVAL_OUTPUT/$panel.out" 2>"$H2G_DOCVAL_OUTPUT/$panel.err" &
  pids+=("$!")
done
failure=0
for pid in "${pids[@]}"; do wait "$pid" || failure=1; done
exit "$failure"
