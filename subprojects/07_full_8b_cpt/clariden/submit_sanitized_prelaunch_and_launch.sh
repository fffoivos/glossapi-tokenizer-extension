#!/usr/bin/env bash
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_STAGE_ROOT:?set completed sanitized D0 stage}"
: "${FULL8_PRELAUNCH_ROOT:?set new prelaunch root}"
: "${FULL8_PRODUCTION_RUN_ROOT:?set new production run root}"
: "${FULL8_INITIAL_MEGATRON:?set canonical TP2 initialization root}"
: "${FULL8_BENCHMARK_ROOT:?set completed sanitized benchmark root}"
: "${FULL8_HF_VISIBILITY_SOURCE:?set completed public-dataset receipt}"
: "${FULL8_LAUNCH_AUTHORIZATION:?set explicit launch authorization}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set pinned Clariden leaf switch for training}"
DRY_RUN=${DRY_RUN:-1}
[[ "$DRY_RUN" == 0 || "$DRY_RUN" == 1 ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "$FULL8_LAUNCH_AUTHORIZATION" == APERTUS8B_FULL_MIXED_CPT ]] || { echo "invalid launch authorization" >&2; exit 2; }

recipe=${FULL8_RECIPE:-$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json}
profiles=${FULL8_PROFILES:-$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json}
selected="$FULL8_BENCHMARK_ROOT/selected_execution_profile.json"
tokenizer=${FULL8_TOKENIZER_ROOT:-/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992}
initial_hf_source=${FULL8_INITIAL_HF_SOURCE:-/capstor/scratch/cscs/fffoivos/models/greek-cpt25b-init-roundtrip/20260731T124000Z-cpt25b-v1/hf_roundtrip}
# The corrected HF view must remain on capstor: its model files are hard-linked
# to the verified zero-drift source, which is also on capstor. Prelaunch logs
# and small receipts remain under FULL8_PRELAUNCH_ROOT on iopsstor.
prelaunch_name=$(basename "$FULL8_PRELAUNCH_ROOT")
default_initial_hf_anchor="/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/_preflight/$prelaunch_name/initial_hf_anchor"
initial_hf_anchor_root=${FULL8_INITIAL_HF_ANCHOR_ROOT:-$default_initial_hf_anchor}
initial_hf_root="$initial_hf_anchor_root/model"
initial_hf_receipt="$initial_hf_anchor_root/receipt.json"
initial_checkpoint_receipt="$FULL8_PRELAUNCH_ROOT/initial_checkpoint_tree.json"
initial_validation_receipt="$FULL8_PRELAUNCH_ROOT/initial_source_validation/initial_validation/initial_validation_receipt.json"
initial_greek_receipt="$FULL8_PRELAUNCH_ROOT/initial_greekmmlu/initial_greekmmlu_receipt.json"
per_document_root="$FULL8_PRELAUNCH_ROOT/per_document_initial"
packed_integrity="$FULL8_PRELAUNCH_ROOT/packed_payload_integrity.json"
nested_root="$FULL8_PRELAUNCH_ROOT/nested_sbatch"
nested_proof="$nested_root/nested_sbatch_proof.json"
conversion_receipt="$FULL8_PRELAUNCH_ROOT/conversion_smoke/conversion_smoke_receipt.json"
smoke_root="$FULL8_PRELAUNCH_ROOT/graceful_stop_smoke"
graceful_receipt="$smoke_root/graceful_stop_smoke_receipt.json"

python3 - "$recipe" "$profiles" "$selected" <<'PY'
import json,sys
from pathlib import Path
r=json.load(open(sys.argv[1])); p=json.load(open(sys.argv[2])); s=json.load(open(sys.argv[3]))
if r.get("recipe_id")!="full8b-mixed-79-20-1-wsd10-sanitized-v1": raise SystemExit("not the sanitized recipe")
if p.get("scientific_recipe_id")!=r["recipe_id"]: raise SystemExit("profile/recipe drift")
if s.get("status")!="frozen" or Path(s.get("recipe",{}).get("path","")).resolve()!=Path(sys.argv[1]).resolve(): raise SystemExit("selected profile drift")
PY

submit() {
  if [[ "$DRY_RUN" == 1 ]]; then
    { printf 'sbatch'; printf ' %q' "$@"; printf '\n'; } >&2
    printf 'DRY_JOB_%s\n' "$(printf '%s\0' "$@" | cksum | awk '{print $1}')"
  else
    sbatch --parsable "$@"
  fi
}

if [[ "$DRY_RUN" == 0 ]]; then
  [[ ! -e "$FULL8_PRELAUNCH_ROOT" && ! -e "$FULL8_PRODUCTION_RUN_ROOT" && ! -e "$initial_hf_anchor_root" ]] || {
    echo "prelaunch, production, or corrected HF anchor root already exists" >&2; exit 2;
  }
  mkdir -p "$FULL8_PRELAUNCH_ROOT/logs"
  python3 - "$FULL8_HF_VISIBILITY_SOURCE" "$FULL8_PRELAUNCH_ROOT/hf_visibility.json" <<'PY'
import shutil,sys
shutil.copy2(sys.argv[1],sys.argv[2])
PY
fi

common="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_STAGE_ROOT=$FULL8_STAGE_ROOT,FULL8_RECIPE=$recipe,FULL8_PROFILES=$profiles"
nested=$(submit \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_NESTED_SMOKE_ROOT=$nested_root" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/prove_nested_sbatch.sbatch")
nested_wait=$(submit --dependency="afterok:$nested" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_NESTED_SBATCH_PROOF=$nested_proof" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/wait_nested_sbatch_proof.sbatch")
freeze_init=$(submit \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_CHECKPOINT_RECEIPT=$initial_checkpoint_receipt" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/freeze_initial_checkpoint.sbatch")
materialize_hf=$(submit \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_INITIAL_HF_SOURCE=$initial_hf_source,FULL8_INITIAL_HF_ROOT=$initial_hf_root,FULL8_INITIAL_HF_RECEIPT=$initial_hf_receipt" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/materialize_corrected_initial_hf.sbatch")
first_end=$(python3 - "$selected" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["selection"]["segment_boundaries"][1])
PY
)
nodes=$(python3 - "$selected" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["selection"]["nodes"])
PY
)
profile_id=$(python3 - "$selected" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["selection"]["profile_id"])
PY
)
train_exclude=$("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$FULL8_TRAIN_LEAF_SWITCH" "$nodes")
source_validation=$(submit --switches=1 --exclude="$train_exclude" --nodes="$nodes" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_RUN_ROOT=$FULL8_PRELAUNCH_ROOT/initial_source_validation,FULL8_START_ITERATION=0,FULL8_END_ITERATION=$first_end,FULL8_LOAD_CHECKPOINT=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_VALIDATION_ONLY=1,FULL8_EXECUTION_PROFILE=$profile_id" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/train_segment.sbatch")
greek=$(submit --dependency="afterok:$materialize_hf" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_INITIAL_HF=$initial_hf_root,FULL8_INITIAL_EVAL_ROOT=$FULL8_PRELAUNCH_ROOT/initial_greekmmlu" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_initial_greekmmlu.sbatch")
doc=$(submit --dependency="afterok:$materialize_hf" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%A_%a.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%A_%a.err" \
  --export="$common,FULL8_VALIDATION_MANIFEST=$FULL8_STAGE_ROOT/validation/validation_manifest.json,FULL8_HF_MODEL=$initial_hf_root,FULL8_HF_TOKENIZER=$tokenizer,FULL8_DOCVAL_OUTPUT=$per_document_root" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/run_per_document_group.sbatch")
packed=$(submit \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_PACKED_INTEGRITY_OUTPUT=$packed_integrity" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/verify_packed_payload_hashes.sbatch")

conversion=$(DRY_RUN="$DRY_RUN" FULL8_CODE_ROOT="$FULL8_CODE_ROOT" FULL8_CODE_BUNDLE_RECEIPT="$FULL8_CODE_BUNDLE_RECEIPT" \
  FULL8_BENCHMARK_ROOT="$FULL8_BENCHMARK_ROOT" FULL8_PRELAUNCH_ROOT="$FULL8_PRELAUNCH_ROOT" \
  FULL8_SELECTED_PROFILE="$selected" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/submit_conversion_smoke.sh")
conversion_gate=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["gate"])' <<<"$conversion")
graceful=$(DRY_RUN="$DRY_RUN" FULL8_CODE_ROOT="$FULL8_CODE_ROOT" FULL8_CODE_BUNDLE_RECEIPT="$FULL8_CODE_BUNDLE_RECEIPT" \
  FULL8_STAGE_ROOT="$FULL8_STAGE_ROOT" FULL8_INITIAL_MEGATRON="$FULL8_INITIAL_MEGATRON" \
  FULL8_SMOKE_RUN_ROOT="$smoke_root" FULL8_SELECTED_PROFILE="$selected" \
  FULL8_TRAIN_LEAF_SWITCH="$FULL8_TRAIN_LEAF_SWITCH" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/submit_graceful_stop_smoke.sh")
graceful_gate=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["resume_coordinator"])' <<<"$graceful")
graceful_wait=$(submit --dependency="afterok:$graceful_gate" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="ALL,FULL8_GRACEFUL_STOP_SMOKE=$graceful_receipt" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/wait_graceful_stop_receipt.sbatch")

handoff_dependency="afterok:$source_validation:$greek:$doc:$freeze_init:$packed:$nested_wait:$conversion_gate:$graceful_wait"
handoff=$(submit --dependency="$handoff_dependency" \
  --output="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.out" --error="$FULL8_PRELAUNCH_ROOT/logs/%x-%j.err" \
  --export="$common,FULL8_PRELAUNCH_ROOT=$FULL8_PRELAUNCH_ROOT,FULL8_BENCHMARK_ROOT=$FULL8_BENCHMARK_ROOT,FULL8_INITIAL_MEGATRON=$FULL8_INITIAL_MEGATRON,FULL8_INITIAL_CHECKPOINT_RECEIPT=$initial_checkpoint_receipt,FULL8_INITIAL_HF_RECEIPT=$initial_hf_receipt,FULL8_INITIAL_VALIDATION_RECEIPT=$initial_validation_receipt,FULL8_INITIAL_GREEKMMLU_RECEIPT=$initial_greek_receipt,FULL8_INITIAL_PER_DOCUMENT_ROOT=$per_document_root,FULL8_PACKED_PAYLOAD_INTEGRITY=$packed_integrity,FULL8_CONVERSION_SMOKE_RECEIPT=$conversion_receipt,FULL8_GRACEFUL_STOP_SMOKE=$graceful_receipt,FULL8_NESTED_SBATCH_PROOF=$nested_proof,FULL8_RUN_ROOT=$FULL8_PRODUCTION_RUN_ROOT,FULL8_LAUNCH_AUTHORIZATION=$FULL8_LAUNCH_AUTHORIZATION,FULL8_TRAIN_LEAF_SWITCH=$FULL8_TRAIN_LEAF_SWITCH" \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/finalize_and_submit_production.sbatch")

graph=$(python3 - "$nested" "$nested_wait" "$freeze_init" "$materialize_hf" "$source_validation" "$greek" "$doc" "$packed" "$conversion_gate" "$graceful_gate" "$graceful_wait" "$handoff" <<'PY'
import json,sys
names=("nested_sbatch_parent","nested_sbatch_proof_waiter","freeze_initial_checkpoint","materialize_corrected_initial_hf","initial_source_validation","initial_greekmmlu","initial_per_document","packed_payload_integrity","conversion_smoke_gate","graceful_stop_resume_coordinator","graceful_stop_receipt_waiter","launch_gate_and_production_handoff")
print(json.dumps(dict(zip(names,sys.argv[1:])),separators=(",",":")))
PY
)
if [[ "$DRY_RUN" == 0 ]]; then
  python3 - "$FULL8_PRELAUNCH_ROOT/submission.json" "$graph" <<'PY'
import datetime,json,os,sys,tempfile
out,raw=sys.argv[1:]
value={"schema_version":"apertus_full_8b_sanitized_prelaunch_submission_v1","status":"submitted","submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"jobs":json.loads(raw)}
fd,tmp=tempfile.mkstemp(prefix=".submission.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as handle:
    json.dump(value,handle,indent=2,sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
os.replace(tmp,out)
PY
fi
printf '%s\n' "$graph"
