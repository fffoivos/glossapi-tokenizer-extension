#!/usr/bin/env bash
set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

: "${EVALUATION_BUNDLE:?set immutable evaluation bundle}"
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${MEGATRON_DIR:?set pinned SwissAI Megatron root}"
: "${TOKENIZER_DIR:?set frozen extended tokenizer}"
: "${PYTHON_COMPAT_DIR:?set frozen NumPy compatibility shim directory}"
: "${CHECKPOINT_EVALUATION_PLAN:?set frozen checkpoint evaluation plan}"
: "${SOURCE_CHECKPOINT_ROOT:?set training checkpoint root}"
: "${SOURCE_ITERATION:?set checkpoint iteration}"
: "${ARM_ID:?set exact D0-D4 arm id}"
: "${EVAL_OUTPUT_ROOT:?set unique checkpoint evaluation output root}"
: "${GREEKMMLU_CLEAN_SUBSET:?set frozen decontaminated GreekMMLU subset}"
: "${CAMPAIGN_MANIFEST:?set frozen campaign manifest}"

"$HOST_PYTHON" - "$CHECKPOINT_EVALUATION_PLAN" "$ARM_ID" "$SOURCE_ITERATION" <<'PY'
import json,sys
path,arm,iteration=sys.argv[1:]
plan=json.load(open(path))
if plan.get("schema_version") != "apertus_mini_checkpoint_evaluation_plan_v1" or plan.get("status") != "frozen":
    raise SystemExit("checkpoint evaluation plan is not frozen")
if arm not in {"D0_mixed","D1_hard_h_to_g","D2_hard_g_to_h","D3_gradual_h_to_g","D4_gradual_g_to_h"}:
    raise SystemExit(f"unknown schedule arm: {arm}")
steps={int(row["iteration"]) for row in plan["checkpoint_rows"] if row["native_greekmmlu_required"]}
if int(iteration) not in steps:
    raise SystemExit(f"iteration {iteration} is not a required native-GreekMMLU checkpoint")
dataset=plan.get("greekmmlu_dataset",{})
expected=("dascim/GreekMMLU","6a03aa06b68beb932fb75edff3a34e50b3674649","All","test")
observed=(dataset.get("repo_id"),dataset.get("revision"),dataset.get("config"),dataset.get("split"))
if observed != expected:
    raise SystemExit(f"GreekMMLU dataset contract drift: {observed}")
PY

[[ ! -e "$EVAL_OUTPUT_ROOT" ]] || {
  echo "refusing to overwrite checkpoint evaluation root: $EVAL_OUTPUT_ROOT" >&2
  exit 2
}
mkdir -p "$EVAL_OUTPUT_ROOT"

model_label="${ARM_ID}_iter${SOURCE_ITERATION}"
export_root="$EVAL_OUTPUT_ROOT/export"
greekmmlu_root="$EVAL_OUTPUT_ROOT/greekmmlu"
output_receipt="$EVAL_OUTPUT_ROOT/exact_checkpoint_native_greekmmlu_receipt.json"

if [[ "$SOURCE_ITERATION" == 0 ]]; then
  "$HOST_PYTHON" \
    "$EVALUATION_BUNDLE/evaluation/prepare_initial_checkpoint_hf_export.py" \
    --campaign-manifest "$CAMPAIGN_MANIFEST" \
    --source-checkpoint-root "$SOURCE_CHECKPOINT_ROOT" \
    --output-root "$export_root" \
    --tokenizer-dir "$TOKENIZER_DIR"
else
  env \
    EVALUATION_BUNDLE="$EVALUATION_BUNDLE" MEGATRON_DIR="$MEGATRON_DIR" \
    SOURCE_CHECKPOINT_ROOT="$SOURCE_CHECKPOINT_ROOT" SOURCE_ITERATION="$SOURCE_ITERATION" \
    TOKENIZER_DIR="$TOKENIZER_DIR" EXPORT_ROOT="$export_root" \
    PYTHON_COMPAT_DIR="$PYTHON_COMPAT_DIR" \
    bash "$EVALUATION_BUNDLE/clariden/convert_checkpoint_for_native_greekmmlu.sbatch"
fi

env \
  NATIVE_GREEK_EVAL_ROOT="$NATIVE_GREEK_EVAL_ROOT" EXPORT_ROOT="$export_root" \
  GREEKMMLU_ROOT="$greekmmlu_root" MODEL_LABEL="$model_label" \
  bash "$EVALUATION_BUNDLE/clariden/run_checkpoint_native_greekmmlu.sbatch"

env \
  EVALUATION_BUNDLE="$EVALUATION_BUNDLE" \
  EXPORT_RECEIPT="$export_root/checkpoint_eval_export_receipt.json" \
  GREEKMMLU_ROOT="$greekmmlu_root" MODEL_LABEL="$model_label" \
  GREEKMMLU_CLEAN_SUBSET="$GREEKMMLU_CLEAN_SUBSET" \
  OUTPUT_RECEIPT="$output_receipt" \
  bash "$EVALUATION_BUNDLE/clariden/finalize_checkpoint_greekmmlu.sbatch"

"$HOST_PYTHON" - "$EVAL_OUTPUT_ROOT/pipeline_state.json" "$ARM_ID" "$SOURCE_ITERATION" "$output_receipt" <<'PY'
import json,sys
out,arm,iteration,receipt=sys.argv[1:]
payload={
    "schema_version":"apertus_mini_checkpoint_native_greekmmlu_pipeline_state_v1",
    "status":"complete",
    "arm_id":arm,
    "source_iteration":int(iteration),
    "receipt":receipt,
}
open(out,"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
