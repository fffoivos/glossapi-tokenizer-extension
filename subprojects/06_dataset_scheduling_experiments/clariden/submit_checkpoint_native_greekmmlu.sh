#!/usr/bin/env bash
set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

: "${EVALUATION_BUNDLE:?set immutable evaluation orchestration bundle}"
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${MEGATRON_DIR:?set pinned SwissAI Megatron root}"
: "${TOKENIZER_DIR:?set frozen extended tokenizer}"
: "${PYTHON_COMPAT_DIR:?set frozen NumPy compatibility shim directory}"
: "${CHECKPOINT_EVALUATION_PLAN:?set frozen checkpoint evaluation plan}"
: "${SOURCE_CHECKPOINT_ROOT:?set training checkpoint root}"
: "${SOURCE_ITERATION:?set checkpoint iteration}"
: "${ARM_ID:?set exact D0-D4 arm id}"
: "${EVAL_OUTPUT_ROOT:?set unique checkpoint evaluation output root}"

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
mkdir -p "$(dirname "$EVAL_OUTPUT_ROOT")"

model_label="${ARM_ID}_iter${SOURCE_ITERATION}"
export_root="$EVAL_OUTPUT_ROOT/export"
greekmmlu_root="$EVAL_OUTPUT_ROOT/greekmmlu"
output_receipt="$EVAL_OUTPUT_ROOT/exact_checkpoint_native_greekmmlu_receipt.json"

convert_job=$(sbatch --parsable \
  --export="ALL,EVALUATION_BUNDLE=$EVALUATION_BUNDLE,MEGATRON_DIR=$MEGATRON_DIR,SOURCE_CHECKPOINT_ROOT=$SOURCE_CHECKPOINT_ROOT,SOURCE_ITERATION=$SOURCE_ITERATION,TOKENIZER_DIR=$TOKENIZER_DIR,EXPORT_ROOT=$export_root,PYTHON_COMPAT_DIR=$PYTHON_COMPAT_DIR" \
  "$EVALUATION_BUNDLE/clariden/convert_checkpoint_for_native_greekmmlu.sbatch")

eval_job=$(sbatch --parsable --dependency="afterok:$convert_job" \
  --export="ALL,NATIVE_GREEK_EVAL_ROOT=$NATIVE_GREEK_EVAL_ROOT,EXPORT_ROOT=$export_root,GREEKMMLU_ROOT=$greekmmlu_root,MODEL_LABEL=$model_label" \
  "$EVALUATION_BUNDLE/clariden/run_checkpoint_native_greekmmlu.sbatch")

receipt_job=$(sbatch --parsable --dependency="afterok:$eval_job" \
  --export="ALL,EVALUATION_BUNDLE=$EVALUATION_BUNDLE,EXPORT_RECEIPT=$export_root/checkpoint_eval_export_receipt.json,GREEKMMLU_ROOT=$greekmmlu_root,MODEL_LABEL=$model_label,OUTPUT_RECEIPT=$output_receipt" \
  "$EVALUATION_BUNDLE/clariden/finalize_checkpoint_greekmmlu.sbatch")

mkdir -p "$EVAL_OUTPUT_ROOT"
"$HOST_PYTHON" - "$EVAL_OUTPUT_ROOT/submission.json" "$CHECKPOINT_EVALUATION_PLAN" "$ARM_ID" "$SOURCE_ITERATION" "$convert_job" "$eval_job" "$receipt_job" <<'PY'
import json,sys
out,plan,arm,iteration,convert_job,eval_job,receipt_job=sys.argv[1:]
payload={
  "schema_version":"apertus_mini_checkpoint_native_greekmmlu_submission_v1",
  "checkpoint_evaluation_plan":plan,
  "arm_id":arm,
  "source_iteration":int(iteration),
  "jobs":{"conversion":convert_job,"native_greekmmlu":eval_job,"receipt":receipt_job},
}
open(out,"w").write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

printf 'conversion=%s\nnative_greekmmlu=%s\nreceipt=%s\n' "$convert_job" "$eval_job" "$receipt_job"
