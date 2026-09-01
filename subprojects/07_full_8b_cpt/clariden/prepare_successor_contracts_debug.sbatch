#!/usr/bin/env bash
#SBATCH --job-name=full8b-successor-contracts-debug
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

# Build a versioned successor contract directory only.  The verified payload,
# plan, pool and validation manifest already exist and are never rewritten.
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_PARENT_STAGE:?set completed parent stage}"
: "${FULL8_STAGE_ROOT:?set verified successor stage}"
: "${FULL8_CONTRACT_ROOT:?set new versioned contract directory under successor stage}"
: "${FULL8_OLD_SELECTED_PROFILE:?set passing parent selected profile}"
: "${FULL8_PROMOTION_RECEIPT:?set passing DP32 promotion receipt}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "successor contract preparation must run on one debug node" >&2
  exit 2
}
case "$FULL8_CONTRACT_ROOT" in
  "$FULL8_STAGE_ROOT"/contracts/*) ;;
  *) echo "contract root must be a versioned child of successor contracts" >&2; exit 2 ;;
esac

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

identity="$FULL8_STAGE_ROOT/inventory/successor_stage_identity.json"
recipe="$FULL8_CONTRACT_ROOT/recipe_8b_full_mixed.sanitized.json"
profiles="$FULL8_CONTRACT_ROOT/execution_profiles.sanitized.json"
selected="$FULL8_CONTRACT_ROOT/selected_execution_profile.json"
receipt="$FULL8_CONTRACT_ROOT/successor_contract_preparation_receipt.json"
for output in "$recipe" "$profiles" "$selected" "$receipt"; do
  [[ ! -e "$output" ]] || { echo "refusing to overwrite $output" >&2; exit 2; }
done

/usr/bin/python3.11 - "$identity" "$FULL8_PARENT_STAGE" "$FULL8_STAGE_ROOT" <<'PY'
import hashlib,json,os,sys
p,parent,successor=sys.argv[1:]
raw=open(p,"rb").read(); value=json.loads(raw)
if not (
 value.get("schema_version")=="apertus_full_8b_successor_stage_identity_v1"
 and value.get("status")=="passed"
 and all(value.get("checks",{}).values())
 and os.path.realpath(value.get("parent_stage",""))==os.path.realpath(parent)
 and os.path.realpath(value.get("successor_stage",""))==os.path.realpath(successor)
): raise SystemExit("successor stage identity drift")
print(json.dumps({"identity_sha256":hashlib.sha256(raw).hexdigest(),"ok":True},sort_keys=True))
PY

mkdir -p "$FULL8_CONTRACT_ROOT"
/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/derive_sanitized_contracts.py" \
  --base-recipe "$FULL8_PARENT_STAGE/contracts/recipe_8b_full_mixed.sanitized.json" \
  --base-profiles "$FULL8_PARENT_STAGE/contracts/execution_profiles.sanitized.json" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --recipe-output "$recipe" --profiles-output "$profiles"
/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/validate_recipe.py" \
  --recipe "$recipe" --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --schedule-manifest "$FULL8_STAGE_ROOT/schedules/schedule_manifest.json"
/usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/rebind_selected_execution_profile.py" \
  --code-root "$FULL8_CODE_ROOT" \
  --old-selected "$FULL8_OLD_SELECTED_PROFILE" \
  --promotion-receipt "$FULL8_PROMOTION_RECEIPT" \
  --recipe "$recipe" --profiles "$profiles" --stage-identity "$identity" \
  --output "$selected"

/usr/bin/python3.11 - "$receipt" "$FULL8_PARENT_STAGE" "$FULL8_STAGE_ROOT" \
  "$identity" "$recipe" "$profiles" "$selected" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,parent,successor,*paths=sys.argv[1:]
def bind(path):
 raw=open(path,"rb").read()
 return {"path":os.path.realpath(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
value={
 "schema_version":"apertus_full_8b_successor_contract_preparation_v1",
 "status":"passed","completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "parent_stage":os.path.realpath(parent),"successor_stage":os.path.realpath(successor),
 "stage_identity":bind(paths[0]),"recipe":bind(paths[1]),
 "profiles":bind(paths[2]),"selected_profile":bind(paths[3]),
 "policy":"versioned_contract_outputs_only_no_payload_or_manifest_mutation_v1",
}
fd,tmp=tempfile.mkstemp(prefix=".successor-contracts.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
print(json.dumps({"ok":True,"recipe":value["recipe"]["sha256"],"selected":value["selected_profile"]["sha256"]},sort_keys=True))
PY
