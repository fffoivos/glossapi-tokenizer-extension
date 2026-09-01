#!/usr/bin/env bash
set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}

: "${EVALUATION_BUNDLE:?set immutable evaluation bundle}"
: "${NATIVE_GREEK_EVAL_ROOT:?set immutable native-Greek evaluator root}"
: "${MEGATRON_DIR:?set pinned SwissAI Megatron root}"
: "${TOKENIZER_DIR:?set frozen extended tokenizer}"
: "${PYTHON_COMPAT_DIR:?set frozen NumPy compatibility shim directory}"
: "${CHECKPOINT_EVALUATION_PLAN:?set frozen checkpoint evaluation plan}"
: "${WAVE_MANIFEST:?set frozen one-to-five checkpoint wave manifest}"
: "${WAVE_OUTPUT_ROOT:?set unique wave log root}"
: "${GREEKMMLU_CLEAN_SUBSET:?set frozen decontaminated GreekMMLU subset}"

"$HOST_PYTHON" - "$WAVE_MANIFEST" "$CHECKPOINT_EVALUATION_PLAN" <<'PY'
import hashlib,json,sys
wave_path,plan_path=sys.argv[1:]
wave=json.load(open(wave_path))
if wave.get("schema_version") != "apertus_mini_greekmmlu_wave_v1" or wave.get("status") != "frozen":
    raise SystemExit("GreekMMLU wave manifest is not frozen")
actual=hashlib.sha256(open(plan_path,"rb").read()).hexdigest()
if wave.get("checkpoint_evaluation_plan_sha256") != actual:
    raise SystemExit("wave manifest does not bind the supplied checkpoint evaluation plan")
PY

[[ ! -e "$WAVE_OUTPUT_ROOT" ]] || {
  echo "refusing to overwrite wave output root: $WAVE_OUTPUT_ROOT" >&2
  exit 2
}
mkdir -p "$WAVE_OUTPUT_ROOT"

job=$(sbatch --parsable \
  --output="$WAVE_OUTPUT_ROOT/slurm-%j.out" \
  --error="$WAVE_OUTPUT_ROOT/slurm-%j.err" \
  --export="ALL,EVALUATION_BUNDLE=$EVALUATION_BUNDLE,NATIVE_GREEK_EVAL_ROOT=$NATIVE_GREEK_EVAL_ROOT,MEGATRON_DIR=$MEGATRON_DIR,TOKENIZER_DIR=$TOKENIZER_DIR,PYTHON_COMPAT_DIR=$PYTHON_COMPAT_DIR,CHECKPOINT_EVALUATION_PLAN=$CHECKPOINT_EVALUATION_PLAN,GREEKMMLU_CLEAN_SUBSET=$GREEKMMLU_CLEAN_SUBSET,WAVE_MANIFEST=$WAVE_MANIFEST,WAVE_OUTPUT_ROOT=$WAVE_OUTPUT_ROOT" \
  "$EVALUATION_BUNDLE/clariden/run_checkpoint_native_greekmmlu_wave.sbatch")
printf 'native_greekmmlu_wave=%s\n' "$job"
