#!/usr/bin/env bash
#SBATCH --job-name=full8b-docval
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=03:00:00
#SBATCH --array=0-3
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_VALIDATION_MANIFEST:?set frozen 13-panel validation manifest}"
: "${FULL8_HF_MODEL:?set one HF model/checkpoint path}"
: "${FULL8_HF_TOKENIZER:?set tokenizer path}"
: "${FULL8_DOCVAL_OUTPUT:?set checkpoint-specific output directory}"
GROUP=${SLURM_ARRAY_TASK_ID}
mkdir -p "$FULL8_DOCVAL_OUTPUT"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific

mapfile -t rows < <(python3 - "$FULL8_VALIDATION_MANIFEST" "$GROUP" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); group=int(sys.argv[2])
for index in range(group*4,min(group*4+4,len(d["panels"]))):
 p=d["panels"][index]
 print(f"{index}\t{p['name']}\t{p['raw_jsonl']['path']}")
PY
)
pids=()
for local_rank in "${!rows[@]}"; do
  IFS=$'\t' read -r panel_index panel_name input <<<"${rows[$local_rank]}"
  output="$FULL8_DOCVAL_OUTPUT/$panel_name.documents.jsonl"
  receipt="$FULL8_DOCVAL_OUTPUT/$panel_name.receipt.json"
  CUDA_VISIBLE_DEVICES=$local_rank uenv run pytorch/v2.9.1:v2 --view=default -- \
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
