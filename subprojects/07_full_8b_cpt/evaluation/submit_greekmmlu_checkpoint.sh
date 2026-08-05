#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_RUN_ROOT:?set run root}"
: "${FULL8_STAGE_ROOT:?set frozen data root}"
: "${FULL8_ITERATION:?set exact checkpoint iteration}"
: "${FULL8_DEPENDENCY:?set afterok dependency for the segment producing this checkpoint}"

RECIPE="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/recipe_8b_full_mixed.json"
EVALUATION_BUNDLE="$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments"
NATIVE_ROOT="$FULL8_CODE_ROOT/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval"
MEGATRON=${FULL8_MEGATRON_ROOT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1}
TOKENIZER=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}
COMPAT=${FULL8_PYTHON_COMPAT_DIR:-/iopsstor/scratch/cscs/fffoivos/orchestration/apertus-cscs-efficiency/20260802T230000Z-mini-b2-v8/compat}
CLEAN_SUBSET=${FULL8_GREEKMMLU_CLEAN_SUBSET:-/capstor/scratch/cscs/fffoivos/cpt_runs/dataset-scheduling-0p5b/20260803T064000Z-static-prelaunch-v2/greekmmlu_clean_subset_manifest.json}
TOKENIZER_SHA=bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b
DRY_RUN=${DRY_RUN:-1}
attempt=${FULL8_EVAL_ATTEMPT:-0}
[[ "$attempt" =~ ^[0-2]$ ]] || { echo "evaluation attempt must be 0, 1, or 2" >&2; exit 2; }

python3 - "$RECIPE" "$FULL8_ITERATION" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); i=int(sys.argv[2])
allowed=set(r["evaluation"]["greekmmlu"]["checkpoint_updates"])-{0}
if i not in allowed: raise SystemExit(f"iteration {i} is not a frozen GreekMMLU milestone")
PY

padded=$(printf '%07d' "$FULL8_ITERATION")
iteration_root="$FULL8_RUN_ROOT/checkpoint_evaluations/iter_$padded"
root="$iteration_root/attempt_$attempt"
export_root="$root/export"
eval_root="$root/greekmmlu"
receipt="$root/exact_checkpoint_native_greekmmlu_receipt.json"
submission="$root/submission.json"
[[ ! -e "$root" ]] || { echo "refusing to overwrite evaluation root: $root" >&2; exit 2; }
namespace="full8b_mixed_iter${FULL8_ITERATION}"
label="full8b_mixed_iter${FULL8_ITERATION}"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'GreekMMLU dry run: iteration=%s dependency=%s output=%s\n' "$FULL8_ITERATION" "$FULL8_DEPENDENCY" "$root"
  exit 0
fi
mkdir -p "$root" "$FULL8_RUN_ROOT/checkpoint_evaluation_logs"

convert=$(sbatch --parsable --dependency="$FULL8_DEPENDENCY" \
  --output="$FULL8_RUN_ROOT/checkpoint_evaluation_logs/%x-%j.out" \
  --error="$FULL8_RUN_ROOT/checkpoint_evaluation_logs/%x-%j.err" \
  --export="ALL,EVALUATION_BUNDLE=$EVALUATION_BUNDLE,MEGATRON_DIR=$MEGATRON,SOURCE_CHECKPOINT_ROOT=$FULL8_RUN_ROOT/checkpoints,SOURCE_ITERATION=$FULL8_ITERATION,TOKENIZER_DIR=$TOKENIZER,EXPORT_ROOT=$export_root,PYTHON_COMPAT_DIR=$COMPAT,TOKENIZER_SHA=$TOKENIZER_SHA" \
  "$EVALUATION_BUNDLE/clariden/convert_checkpoint_for_native_greekmmlu.sbatch")
evaluate=$(sbatch --parsable --dependency="afterok:$convert" \
  --output="$FULL8_RUN_ROOT/checkpoint_evaluation_logs/%x-%j.out" \
  --error="$FULL8_RUN_ROOT/checkpoint_evaluation_logs/%x-%j.err" \
  --export="ALL,NATIVE_GREEK_EVAL_ROOT=$NATIVE_ROOT,EXPORT_ROOT=$export_root,GREEKMMLU_ROOT=$eval_root,MODEL_LABEL=$label,EVALUATION_NAMESPACE=$namespace,EVAL_DTYPE=float32" \
  "$EVALUATION_BUNDLE/clariden/run_checkpoint_native_greekmmlu.sbatch")
finalize=$(sbatch --parsable --dependency="afterok:$evaluate" \
  --output="$FULL8_RUN_ROOT/checkpoint_evaluation_logs/%x-%j.out" \
  --error="$FULL8_RUN_ROOT/checkpoint_evaluation_logs/%x-%j.err" \
  --export="ALL,EVALUATION_BUNDLE=$EVALUATION_BUNDLE,EXPORT_RECEIPT=$export_root/checkpoint_eval_export_receipt.json,GREEKMMLU_ROOT=$eval_root,MODEL_LABEL=$label,OUTPUT_RECEIPT=$receipt,GREEKMMLU_CLEAN_SUBSET=$CLEAN_SUBSET,EVALUATION_NAMESPACE=$namespace" \
  "$EVALUATION_BUNDLE/clariden/finalize_checkpoint_greekmmlu.sbatch")
doc_job=""
if [[ "$FULL8_ITERATION" == 15398 || "$FULL8_ITERATION" == 19248 ]]; then
  doc_job=$(sbatch --parsable --dependency="afterok:$convert" \
    --output="$FULL8_RUN_ROOT/checkpoint_evaluation_logs/%x-%j.out" \
    --error="$FULL8_RUN_ROOT/checkpoint_evaluation_logs/%x-%j.err" \
    --export="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_VALIDATION_MANIFEST=$FULL8_STAGE_ROOT/validation/validation_manifest.json,FULL8_HF_MODEL=$export_root/hf,FULL8_HF_TOKENIZER=$TOKENIZER,FULL8_DOCVAL_OUTPUT=$root/per_document" \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_per_document_group.sbatch")
fi
python3 - "$submission" "$FULL8_ITERATION" "$convert" "$evaluate" "$finalize" "$doc_job" "$receipt" <<'PY'
import datetime,json,os,sys,tempfile
out,iteration,convert,evaluate,finalize,doc_job,receipt=sys.argv[1:]
value={"schema_version":"apertus_full_8b_greekmmlu_submission_v1","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"iteration":int(iteration),"attempt":int(os.path.basename(os.path.dirname(out)).split("_")[-1]),"jobs":{"conversion":convert,"greekmmlu":evaluate,"receipt":finalize,"per_document":doc_job or None},"expected_receipt":receipt}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
printf '%s %s %s %s\n' "$convert" "$evaluate" "$finalize" "$doc_job"
