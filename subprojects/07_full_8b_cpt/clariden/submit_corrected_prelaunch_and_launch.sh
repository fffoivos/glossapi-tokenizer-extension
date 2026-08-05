#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_PARENT_STAGE:?set original frozen stage}"
: "${FULL8_CORRECTED_STAGE:?set new corrected validation stage}"
: "${FULL8_PRELAUNCH_ROOT:?set new corrected prelaunch root}"
: "${FULL8_PRODUCTION_RUN_ROOT:?set new production run root}"
: "${FULL8_INITIAL_MEGATRON:?set canonical TP2 initialization root}"
: "${FULL8_BENCHMARK_ROOT:?set completed benchmark root}"
: "${FULL8_CONVERSION_SMOKE_RECEIPT:?set completed conversion smoke}"
: "${FULL8_HF_VISIBILITY_SOURCE:?set completed dataset-visibility receipt}"
: "${FULL8_VALIDATION_PYTHON:?set validation Python environment}"
: "${FULL8_GRACEFUL_STOP_SMOKE:?set completed graceful-stop/restart smoke receipt}"
: "${FULL8_LAUNCH_AUTHORIZATION:?set explicit launch authorization}"
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
selected="$FULL8_BENCHMARK_ROOT/selected_execution_profile.json"
tokenizer=/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992
initial_hf_source=${FULL8_INITIAL_HF_SOURCE:-/capstor/scratch/cscs/fffoivos/models/greek-cpt25b-init-roundtrip/20260731T124000Z-cpt25b-v1/hf_roundtrip}
initial_hf_root="$FULL8_PRELAUNCH_ROOT/initial_hf_anchor/model"
initial_hf_receipt="$FULL8_PRELAUNCH_ROOT/initial_hf_anchor/receipt.json"
initial_checkpoint_receipt="$FULL8_PRELAUNCH_ROOT/initial_checkpoint_tree.json"
initial_validation_receipt="$FULL8_PRELAUNCH_ROOT/initial_source_validation/initial_validation/initial_validation_receipt.json"
initial_greek_receipt="$FULL8_PRELAUNCH_ROOT/initial_greekmmlu/initial_greekmmlu_receipt.json"
per_document_root="$FULL8_PRELAUNCH_ROOT/per_document_initial"
packed_integrity="$FULL8_PRELAUNCH_ROOT/packed_payload_integrity.json"

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    printf 'DRY_JOB_%s\n' "$(printf '%s\0' "$@" | cksum | awk '{print $1}')"
  else
    sbatch --parsable "$@"
  fi
}

if [[ "$DRY_RUN" == 0 ]]; then
  [[ ! -e "$FULL8_CORRECTED_STAGE" && ! -e "$FULL8_PRELAUNCH_ROOT" && ! -e "$FULL8_PRODUCTION_RUN_ROOT" ]] || {
    echo "corrected stage, prelaunch root, or production root already exists" >&2; exit 2;
  }
  mkdir -p "$FULL8_PRELAUNCH_ROOT/logs"
  python3 - "$FULL8_HF_VISIBILITY_SOURCE" "$FULL8_PRELAUNCH_ROOT/hf_visibility.json" <<'PY'
import shutil,sys
shutil.copy2(sys.argv[1],sys.argv[2])
PY
fi

common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT"
stage=$(submit \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_PARENT_STAGE=$FULL8_PARENT_STAGE,FULL8_CORRECTED_STAGE=$FULL8_CORRECTED_STAGE,FULL8_VALIDATION_PYTHON=$FULL8_VALIDATION_PYTHON" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/build_corrected_validation_stage.sbatch")

freeze_init=$(submit \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_CHECKPOINT_RECEIPT=$initial_checkpoint_receipt" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/freeze_initial_checkpoint.sbatch")

materialize_hf=$(submit \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_INITIAL_HF_SOURCE=$initial_hf_source,FULL8_INITIAL_HF_ROOT=$initial_hf_root,FULL8_INITIAL_HF_RECEIPT=$initial_hf_receipt" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/materialize_corrected_initial_hf.sbatch")

source_validation=$(submit --nodes=16 --dependency="afterok:$stage" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_STAGE_ROOT=$FULL8_CORRECTED_STAGE,FULL8_RUN_ROOT=$FULL8_PRELAUNCH_ROOT/initial_source_validation,FULL8_START_ITERATION=0,FULL8_END_ITERATION=3208,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_VALIDATION_ONLY=1" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")

greek=$(submit --dependency="afterok:$materialize_hf" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_INITIAL_HF=$initial_hf_root,FULL8_INITIAL_EVAL_ROOT=$FULL8_PRELAUNCH_ROOT/initial_greekmmlu" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_initial_greekmmlu.sbatch")

doc=$(submit --dependency="afterok:$stage:$materialize_hf" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%A_%a.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%A_%a.err" \
  --export="$common,FULL8_VALIDATION_MANIFEST=$FULL8_CORRECTED_STAGE/validation/validation_manifest.json,FULL8_HF_MODEL=$initial_hf_root,FULL8_HF_TOKENIZER=$tokenizer,FULL8_DOCVAL_OUTPUT=$per_document_root" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_per_document_group.sbatch")

packed=$(submit --dependency="afterok:$stage" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_STAGE_ROOT=$FULL8_CORRECTED_STAGE,FULL8_PACKED_INTEGRITY_OUTPUT=$packed_integrity" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/verify_packed_payload_hashes.sbatch")

handoff_dependency="afterok:$source_validation:$greek:$doc:$freeze_init:$packed"
handoff=$(submit --dependency="$handoff_dependency" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_STAGE_ROOT=$FULL8_CORRECTED_STAGE,FULL8_PRELAUNCH_ROOT=$FULL8_PRELAUNCH_ROOT,FULL8_BENCHMARK_ROOT=$FULL8_BENCHMARK_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_CHECKPOINT_RECEIPT=$initial_checkpoint_receipt,FULL8_INITIAL_HF_RECEIPT=$initial_hf_receipt,FULL8_INITIAL_VALIDATION_RECEIPT=$initial_validation_receipt,FULL8_INITIAL_GREEKMMLU_RECEIPT=$initial_greek_receipt,FULL8_INITIAL_PER_DOCUMENT_ROOT=$per_document_root,FULL8_PACKED_PAYLOAD_INTEGRITY=$packed_integrity,FULL8_CONVERSION_SMOKE_RECEIPT=$FULL8_CONVERSION_SMOKE_RECEIPT,FULL8_GRACEFUL_STOP_SMOKE=$FULL8_GRACEFUL_STOP_SMOKE,FULL8_RUN_ROOT=$FULL8_PRODUCTION_RUN_ROOT,FULL8_LAUNCH_AUTHORIZATION=$FULL8_LAUNCH_AUTHORIZATION" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_and_submit_production.sbatch")

graph=$(python3 - "$stage" "$freeze_init" "$materialize_hf" "$source_validation" "$greek" "$doc" "$packed" "$handoff" <<'PY'
import json,sys
names=("corrected_stage","freeze_initial_checkpoint","materialize_corrected_initial_hf","initial_source_validation","initial_greekmmlu","initial_per_document","packed_payload_integrity","launch_gate_and_production_handoff")
print(json.dumps(dict(zip(names,sys.argv[1:])),separators=(",",":")))
PY
)
if [[ "$DRY_RUN" == 0 ]]; then
  python3 - "$FULL8_PRELAUNCH_ROOT/submission.json" "$graph" <<'PY'
import datetime,json,os,sys,tempfile
out,raw=sys.argv[1:]
value={"schema_version":"apertus_full_8b_corrected_prelaunch_submission_v1","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"jobs":json.loads(raw)}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
fi
printf '%s\n' "$graph"
