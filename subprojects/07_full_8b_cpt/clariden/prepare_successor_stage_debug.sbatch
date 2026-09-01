#!/usr/bin/env bash
#SBATCH --job-name=full8b-successor-stage-debug
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_PARENT_STAGE:?set completed parent stage}"
: "${FULL8_STAGE_ROOT:?set successor stage}"
: "${FULL8_OLD_SELECTED_PROFILE:?set passing parent selected profile}"
: "${FULL8_PROMOTION_RECEIPT:?set passing DP32 promotion receipt}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "successor-stage preparation must run on one debug node" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

identity="$FULL8_STAGE_ROOT/inventory/successor_stage_identity.json"
packed="$FULL8_STAGE_ROOT/inventory/packed_corpus_receipt.json"
schedule="$FULL8_STAGE_ROOT/schedules/schedule_manifest.json"
validation="$FULL8_STAGE_ROOT/validation/validation_manifest.json"
recipe="$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json"
profiles="$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json"
selected="$FULL8_STAGE_ROOT/contracts/selected_execution_profile.json"
receipt="$FULL8_STAGE_ROOT/contracts/successor_stage_preparation_receipt.json"
for output in "$identity" "$recipe" "$profiles" "$selected" "$receipt"; do
  [[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
done
verify_or_copy_exact() {
  local source=$1 destination=$2
  [[ -f "$source" ]] || return 2
  if [[ -e "$destination" ]]; then
    cmp -s "$source" "$destination" || {
      echo "existing successor artifact differs from parent: $destination" >&2
      return 3
    }
    return 0
  fi
  mkdir -p "$(dirname "$destination")"
  cp "$source" "$destination.partial"
  cmp -s "$source" "$destination.partial" || return 3
  mv "$destination.partial" "$destination"
}
verify_or_copy_exact "$FULL8_PARENT_STAGE/inventory/packed_corpus_receipt.json" "$packed"
verify_or_copy_exact "$FULL8_PARENT_STAGE/schedules/schedule_manifest.json" "$schedule"
verify_or_copy_exact "$FULL8_PARENT_STAGE/validation/validation_manifest.json" "$validation"

/usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/compare_successor_stage.py" \
  --parent-stage "$FULL8_PARENT_STAGE" --successor-stage "$FULL8_STAGE_ROOT" \
  --output "$identity"

mkdir -p "$FULL8_STAGE_ROOT/contracts"
/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/derive_sanitized_contracts.py" \
  --base-recipe "$FULL8_PARENT_STAGE/contracts/recipe_8b_full_mixed.sanitized.json" \
  --base-profiles "$FULL8_PARENT_STAGE/contracts/execution_profiles.sanitized.json" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --recipe-output "$recipe" --profiles-output "$profiles"
/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_recipe.py" \
  --recipe "$recipe" --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --schedule-manifest "$schedule"
/usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/rebind_selected_execution_profile.py" \
  --code-root "$FULL8_CODE_ROOT" \
  --old-selected "$FULL8_OLD_SELECTED_PROFILE" \
  --promotion-receipt "$FULL8_PROMOTION_RECEIPT" \
  --recipe "$recipe" --profiles "$profiles" --stage-identity "$identity" \
  --output "$selected"

/usr/bin/python3.11 - "$receipt" "$FULL8_PARENT_STAGE" "$FULL8_STAGE_ROOT" \
  "$identity" "$packed" "$schedule" "$validation" "$recipe" "$profiles" "$selected" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,parent,successor,*paths=sys.argv[1:]
def bind(path):
 raw=open(path,"rb").read()
 return {"path":os.path.realpath(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
value={
 "schema_version":"apertus_full_8b_successor_stage_preparation_v1",
 "status":"passed","completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "parent_stage":os.path.realpath(parent),"successor_stage":os.path.realpath(successor),
 "identity":bind(paths[0]),"packed_receipt":bind(paths[1]),
 "schedule_manifest":bind(paths[2]),"validation_manifest":bind(paths[3]),
 "recipe":bind(paths[4]),"profiles":bind(paths[5]),"selected_profile":bind(paths[6]),
 "copied_artifacts":"exact bytes from completed parent stage; raw validation and packed payload remain parent-bound absolute paths",
}
fd,tmp=tempfile.mkstemp(prefix=".successor.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
print(json.dumps({"ok":True,"recipe":value["recipe"]["sha256"],"selected":value["selected_profile"]["sha256"]},sort_keys=True))
PY
