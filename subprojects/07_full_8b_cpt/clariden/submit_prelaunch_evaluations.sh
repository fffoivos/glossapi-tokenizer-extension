#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set frozen data root}"
: "${FULL8_PRELAUNCH_ROOT:?set new prelaunch output root}"
: "${FULL8_INITIAL_MEGATRON:?set verified TP2 initialization root}"
: "${FULL8_INITIAL_HF:?set verified HF roundtrip initialization root}"
: "${FULL8_BENCHMARK_ROOT:?set completed benchmark root}"
: "${FULL8_SELECTED_PROFILE:?set benchmark-selected profile receipt}"
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 1 || ! -e "$FULL8_PRELAUNCH_ROOT" ]] || { echo "prelaunch root exists" >&2; exit 2; }

common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT"
if [[ "$DRY_RUN" == 1 ]]; then
  printf 'sbatch --export=%q %q\n' "$common,FULL8_RUN_ROOT=$FULL8_PRELAUNCH_ROOT,FULL8_START_ITERATION=0,FULL8_END_ITERATION=3208,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_VALIDATION_ONLY=1" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch"
  printf 'sbatch --export=%q %q\n' "$common,FULL8_INITIAL_HF=$FULL8_INITIAL_HF,FULL8_INITIAL_EVAL_ROOT=$FULL8_PRELAUNCH_ROOT/initial_greekmmlu" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_initial_greekmmlu.sbatch"
  printf 'sbatch --array=0-3 --export=%q %q\n' "$common,FULL8_VALIDATION_MANIFEST=$FULL8_STAGE_ROOT/validation/validation_manifest.json,FULL8_HF_MODEL=$FULL8_INITIAL_HF,FULL8_HF_TOKENIZER=/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992,FULL8_DOCVAL_OUTPUT=$FULL8_PRELAUNCH_ROOT/per_document_initial" "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_per_document_group.sbatch"
  DRY_RUN=1 FULL8_CODE_ROOT="$FULL8_CODE_ROOT" FULL8_CODE_BUNDLE_RECEIPT="$FULL8_CODE_BUNDLE_RECEIPT" FULL8_BENCHMARK_ROOT="$FULL8_BENCHMARK_ROOT" \
    FULL8_PRELAUNCH_ROOT="$FULL8_PRELAUNCH_ROOT" FULL8_SELECTED_PROFILE="$FULL8_SELECTED_PROFILE" \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/submit_conversion_smoke.sh"
  exit 0
fi
mkdir -p "$FULL8_PRELAUNCH_ROOT/logs"
source_job=$(sbatch --parsable \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_RUN_ROOT=$FULL8_PRELAUNCH_ROOT,FULL8_START_ITERATION=0,FULL8_END_ITERATION=3208,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_VALIDATION_ONLY=1" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")
greek_job=$(sbatch --parsable \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_INITIAL_HF=$FULL8_INITIAL_HF,FULL8_INITIAL_EVAL_ROOT=$FULL8_PRELAUNCH_ROOT/initial_greekmmlu" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_initial_greekmmlu.sbatch")
doc_job=$(sbatch --parsable \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%A_%a.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%A_%a.err" \
  --export="$common,FULL8_VALIDATION_MANIFEST=$FULL8_STAGE_ROOT/validation/validation_manifest.json,FULL8_HF_MODEL=$FULL8_INITIAL_HF,FULL8_HF_TOKENIZER=/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992,FULL8_DOCVAL_OUTPUT=$FULL8_PRELAUNCH_ROOT/per_document_initial" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_per_document_group.sbatch")
conversion=$(DRY_RUN=0 FULL8_CODE_ROOT="$FULL8_CODE_ROOT" FULL8_CODE_BUNDLE_RECEIPT="$FULL8_CODE_BUNDLE_RECEIPT" FULL8_BENCHMARK_ROOT="$FULL8_BENCHMARK_ROOT" \
  FULL8_PRELAUNCH_ROOT="$FULL8_PRELAUNCH_ROOT" FULL8_SELECTED_PROFILE="$FULL8_SELECTED_PROFILE" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/submit_conversion_smoke.sh")
python3 - "$FULL8_PRELAUNCH_ROOT/submission.json" "$source_job" "$greek_job" "$doc_job" "$conversion" <<'PY'
import datetime,json,os,sys,tempfile
out,source,greek,doc,conversion=sys.argv[1:]
value={"schema_version":"apertus_full_8b_prelaunch_submission_v2","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"jobs":{"source_validation":source,"initial_greekmmlu":greek,"initial_per_document":doc},"conversion_smoke":json.loads(conversion)}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f: json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
print(json.dumps(value,separators=(",",":")))
PY
