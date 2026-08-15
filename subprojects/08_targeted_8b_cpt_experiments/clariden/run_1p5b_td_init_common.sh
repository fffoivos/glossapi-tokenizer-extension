#!/usr/bin/env bash

# Shared scientific body for the 1.5B Token-Distillation initialization.
# Resource-class wrappers must verify their immutable bundle and allocation
# before entering this file. Keep the TD command identical across wrappers.
set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"

parent="$H2G_STAGE_ROOT/assets/init/1p5b_parent_hf"
tokenizer="$H2G_STAGE_ROOT/assets/tokenizer_148480"
inputs="$H2G_STAGE_ROOT/assets/init/td_inputs"
reference="$H2G_STAGE_ROOT/assets/init/1p5b_retok_reference"
td_model="$H2G_STAGE_ROOT/assets/init/1p5b_td_hf_raw_v2"
reference_receipt="$H2G_STAGE_ROOT/receipts/1p5b_retok_reference.json"
verification="$H2G_STAGE_ROOT/receipts/1p5b_td_initialization_v2.json"
policy="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/1p5b_td_acceptance_policy_v2.json"
policy_authorization="$H2G_STAGE_ROOT/receipts/1p5b_td_policy_authorization.json"
reference_probe="$H2G_STAGE_ROOT/receipts/1p5b_td_objective_reference_v2.json"
td_probe="$H2G_STAGE_ROOT/receipts/1p5b_td_objective_trained_v2.json"
parent_receipt="$H2G_STAGE_ROOT/receipts/1p5b_parent_hf_materialization.json"
tokenizer_receipt="$H2G_STAGE_ROOT/receipts/tokenizer_148480.json"
td_inputs_receipt="$H2G_STAGE_ROOT/receipts/td_training_inputs.json"
for path in "$parent/config.json" "$tokenizer/tokenizer.json" "$inputs/selected_token_ids.txt" "$policy" "$policy_authorization" "$parent_receipt" "$tokenizer_receipt" "$td_inputs_receipt"; do
  [[ -s "$path" ]] || { echo "missing prerequisite: $path" >&2; exit 2; }
done
[[ ! -e "$verification" ]] || { echo "TD verification already exists: $verification"; exit 0; }
if [[ -e "$reference" || -e "$reference_receipt" ]]; then
  [[ -d "$reference" && -s "$reference_receipt" ]] || { echo "partial ReTok reference output exists" >&2; exit 2; }
fi
export H2G_CODE_ROOT H2G_CODE_RECEIPT TOKENIZERS_PARALLELISM=false
if [[ ! -e "$reference_receipt" ]]; then
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/build_retok_reference_init.py" \
    --base-model "$parent" --base-materialization-receipt "$parent_receipt" \
    --extended-tokenizer "$tokenizer" --extended-tokenizer-receipt "$tokenizer_receipt" \
    --tokenizer-compatibility "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/1p5b_tokenizer_compatibility_v1.json" \
    --retok-tool "$H2G_CODE_ROOT/frozen_td_tools/retok.py" \
    --output-root "$reference" --output-receipt "$reference_receipt" \
    --expected-hidden-size 2048 --expected-hidden-layers 16
fi
export PYTHONPATH="$H2G_CODE_ROOT/frozen_td_tools/external/token-distillation:${PYTHONPATH:-}"
if [[ ! -e "$reference_probe" ]]; then
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/evaluate_td_objective.py" \
    --role reference --model "$reference" \
    --td-wrapper "$H2G_CODE_ROOT/frozen_td_tools/train_retok_td.py" \
    --base-tokenizer "$parent" --student-tokenizer "$tokenizer" \
    --coverage-jsonl "$inputs/coverage/td_coverage_prepass.jsonl" \
    --snippets-jsonl "$inputs/coverage/td_snippet_index/snippets.jsonl" \
    --token-ids-file "$inputs/selected_token_ids.txt" --policy "$policy" \
    --output "$reference_probe"
fi
if [[ ! -e "$td_model" ]]; then
  td_work=$(mktemp -d "$H2G_STAGE_ROOT/assets/init/.1p5b_td_hf_raw_v2.XXXXXX")
  cleanup_td_work() { rm -rf -- "$td_work"; }
  trap cleanup_td_work EXIT
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$H2G_CODE_ROOT/frozen_td_tools/train_retok_td.py" \
    --student-model "$reference" --base-tokenizer "$parent" --student-tokenizer "$tokenizer" \
    --coverage-jsonl "$inputs/coverage/td_coverage_prepass.jsonl" \
    --snippets-jsonl "$inputs/coverage/td_snippet_index/snippets.jsonl" \
    --token-ids-file "$inputs/selected_token_ids.txt" --output-dir "$td_work/model" \
    --base-vocab-size 131072 --new-id-start 131072 --new-id-end 148480 \
    --snippets-per-token 25 --min-accepted-snippets-per-token 25 \
    --min-trained-token-fraction 0.99 --epochs 1 --batch-size 8 \
    --learning-rate 1e-4 --target-layer 6 --dtype bfloat16 --device cuda --seed 20260523
  [[ -s "$td_work/model/retok_td_manifest.json" ]] || { echo "TD temporary output is incomplete" >&2; exit 2; }
  mv "$td_work/model" "$td_model"
  rmdir "$td_work"
  trap - EXIT
fi
[[ -s "$td_model/retok_td_manifest.json" ]] || { echo "partial TD output exists" >&2; exit 2; }
if [[ ! -e "$td_probe" ]]; then
  uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
    "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/evaluate_td_objective.py" \
    --role td --model "$td_model" \
    --td-wrapper "$H2G_CODE_ROOT/frozen_td_tools/train_retok_td.py" \
    --base-tokenizer "$parent" --student-tokenizer "$tokenizer" \
    --coverage-jsonl "$inputs/coverage/td_coverage_prepass.jsonl" \
    --snippets-jsonl "$inputs/coverage/td_snippet_index/snippets.jsonl" \
    --token-ids-file "$inputs/selected_token_ids.txt" --policy "$policy" \
    --output "$td_probe"
fi
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/verify_td_initialization.py" \
  --parent-model "$parent" --parent-materialization-receipt "$parent_receipt" \
  --retok-reference "$reference" --retok-reference-receipt "$reference_receipt" --td-model "$td_model" \
  --td-manifest "$td_model/retok_td_manifest.json" --td-training-inputs-receipt "$td_inputs_receipt" \
  --tokenizer-receipt "$tokenizer_receipt" --acceptance-policy "$policy" \
  --policy-authorization "$policy_authorization" \
  --reference-objective-probe "$reference_probe" --td-objective-probe "$td_probe" \
  --tokenizer-sha256 358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394 \
  --output "$verification"
