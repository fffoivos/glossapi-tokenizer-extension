#!/usr/bin/env bash
#SBATCH --job-name=full8b-launch-gate-debug
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_STAGE_ROOT:?set frozen successor data root}"
: "${FULL8_PRELAUNCH_ROOT:?set new prelaunch root}"
: "${FULL8_BENCHMARK_ROOT:?set completed benchmark root}"
: "${FULL8_BENCHMARK_RECEIPT:?set completed DP32 promotion receipt}"
: "${FULL8_SOURCE_SELECTED_PROFILE:?set proven pre-successor selected profile}"
: "${FULL8_SELECTED_PROFILE:?set successor-bound selected profile}"
: "${FULL8_INITIAL_MEGATRON:?set verified TP2 initialization root}"
: "${FULL8_INITIAL_CHECKPOINT_RECEIPT:?set checkpoint-tree receipt}"
: "${FULL8_INITIAL_HF_RECEIPT:?set corrected iteration-zero HF receipt}"
: "${FULL8_INITIAL_VALIDATION_RECEIPT:?set corrected initial validation receipt}"
: "${FULL8_INITIAL_GREEKMMLU_RECEIPT:?set corrected initial GreekMMLU receipt}"
: "${FULL8_INITIAL_PER_DOCUMENT_ROOT:?set initial per-document root}"
: "${FULL8_PACKED_PAYLOAD_INTEGRITY:?set packed-payload hash receipt}"
: "${FULL8_CONVERSION_SMOKE_RECEIPT:?set conversion smoke receipt}"
: "${FULL8_GRACEFUL_STOP_SMOKE:?set graceful-stop receipt}"
: "${FULL8_NESTED_SBATCH_PROOF:?set nested-sbatch proof}"
: "${FULL8_RESOURCE_ROUTING_SMOKE:?set resource-routing smoke receipt}"
: "${FULL8_EVAL_OVERLAP_SMOKE:?set evaluation-overlap smoke receipt}"
: "${FULL8_HF_VISIBILITY:?set public HF visibility receipt}"
: "${FULL8_RUN_ROOT:?set new production run root}"
: "${FULL8_LAUNCH_AUTHORIZATION:?set explicit production authorization}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set pinned Clariden leaf switch}"
[[ "$FULL8_LAUNCH_AUTHORIZATION" == APERTUS8B_FULL_MIXED_CPT ]] || {
  echo "invalid production launch authorization" >&2
  exit 2
}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "launch gate must run on debug" >&2
  exit 2
}

recipe=${FULL8_RECIPE:-$FULL8_STAGE_ROOT/contracts/recipe_8b_full_mixed.sanitized.json}
profiles=${FULL8_PROFILES:-$FULL8_STAGE_ROOT/contracts/execution_profiles.sanitized.json}
successor_identity=${FULL8_SUCCESSOR_STAGE_IDENTITY:-$FULL8_STAGE_ROOT/inventory/successor_stage_identity.json}
owner=${FULL8_OWNER_DECISIONS:-$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/owner_decisions_20260805.json}
environment="$FULL8_PRELAUNCH_ROOT/launch_environment.json"
gate="$FULL8_PRELAUNCH_ROOT/launch_gate.json"
graph="$FULL8_RUN_ROOT/submissions/resource_aware_launch_graph.json"
routing="$FULL8_PRELAUNCH_ROOT/allocation_routing_receipt.json"
ops_gate="$FULL8_PRELAUNCH_ROOT/operational_launch_gate.json"
train_exclude=$("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$FULL8_TRAIN_LEAF_SWITCH" 16)
[[ ! -e "$environment" && ! -e "$gate" && ! -e "$ops_gate" && ! -e "$FULL8_RUN_ROOT" ]] || {
  echo "refusing to overwrite launch evidence or production run root" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

/usr/bin/python3.11 - "$FULL8_RESOURCE_ROUTING_SMOKE" "$FULL8_CODE_ROOT" \
  "$FULL8_CODE_BUNDLE_RECEIPT" "$FULL8_OPS_ROOT" "$FULL8_OPS_BUNDLE_RECEIPT" <<'PY'
import json,os,sys
p,code_root,code_receipt,ops_root,ops_receipt=sys.argv[1:]
d=json.load(open(p))
checks=(
 d.get("schema_version")=="apertus_full_8b_resource_routing_smoke_v1",
 d.get("status")=="passed",
 all(d.get("checks",{}).values()),
 d.get("parent_partition")==d.get("child_partition")=="debug",
 d.get("child_nodes")==1,
 d.get("nested_submit_flag")=="--uenv-passthrough=ignore",
 os.path.realpath(d.get("scientific_root",""))==os.path.realpath(code_root),
 os.path.realpath(d.get("scientific_receipt",""))==os.path.realpath(code_receipt),
 os.path.realpath(d.get("operational_root",""))==os.path.realpath(ops_root),
 os.path.realpath(d.get("operational_receipt",""))==os.path.realpath(ops_receipt),
)
if not all(checks): raise SystemExit("resource-routing smoke drift")
PY

/usr/bin/python3.11 - "$FULL8_EVAL_OVERLAP_SMOKE" "$FULL8_CODE_ROOT" \
  "$FULL8_CODE_BUNDLE_RECEIPT" "$FULL8_OPS_ROOT" "$FULL8_OPS_BUNDLE_RECEIPT" <<'PY'
import json,os,sys
p,code_root,code_receipt,ops_root,ops_receipt=sys.argv[1:]
d=json.load(open(p))
checks=(
 d.get("schema_version")=="apertus_full_8b_evaluation_overlap_smoke_v1",
 d.get("status")=="passed",
 all(d.get("checks",{}).values()),
 d.get("partition")=="debug" and d.get("nodes")==4,
 len(d.get("steps",[]))==5 and all(row.get("exit_code")==0 for row in d["steps"]),
 os.path.realpath(d.get("scientific_root",""))==os.path.realpath(code_root),
 os.path.realpath(d.get("scientific_receipt",""))==os.path.realpath(code_receipt),
 os.path.realpath(d.get("operational_root",""))==os.path.realpath(ops_root),
 os.path.realpath(d.get("operational_receipt",""))==os.path.realpath(ops_receipt),
)
if not all(checks): raise SystemExit("evaluation-overlap smoke drift")
PY

mkdir -p "$FULL8_PRELAUNCH_ROOT"
/usr/bin/python3.11 - "$ops_gate" "$FULL8_CODE_ROOT" "$FULL8_CODE_BUNDLE_RECEIPT" \
  "$FULL8_OPS_ROOT" "$FULL8_OPS_BUNDLE_RECEIPT" \
  "$FULL8_RESOURCE_ROUTING_SMOKE" "$FULL8_EVAL_OVERLAP_SMOKE" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,code_root,code_receipt,ops_root,ops_receipt,routing_smoke,overlap_smoke=sys.argv[1:]
def binding(path):
 path=os.path.realpath(path); raw=open(path,"rb").read()
 return {"path":path,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
value={
 "schema_version":"apertus_full_8b_operational_launch_gate_v1",
 "status":"passed","created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "scientific_root":os.path.realpath(code_root),"scientific_receipt":binding(code_receipt),
 "operational_root":os.path.realpath(ops_root),"operational_receipt":binding(ops_receipt),
 "resource_routing_smoke":binding(routing_smoke),
 "evaluation_overlap_smoke":binding(overlap_smoke),
 "resource_policy":{
  "training":"normal_16node_12h_single_leaf",
  "auxiliary":"debug_1to4node_at_most_90m",
  "debug_qos_chain":"one_running_one_afterany_continuation",
 },
}
fd,tmp=tempfile.mkstemp(prefix=".ops-gate.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY

/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/capture_launch_environment.py" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --leaf-switch "$FULL8_TRAIN_LEAF_SWITCH" \
  --leaf-exclusion "$train_exclude" \
  --storage-path /capstor/scratch/cscs/fffoivos/runs \
  --output "$environment"

/usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/build_successor_launch_gate.py" \
  --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --code-root "$FULL8_CODE_ROOT" \
  --recipe "$recipe" --profiles "$profiles" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --source-selected-profile "$FULL8_SOURCE_SELECTED_PROFILE" \
  --source-promotion-receipt "$FULL8_BENCHMARK_RECEIPT" \
  --stage-identity "$successor_identity" \
  --owner-decisions "$owner" --hf-visibility "$FULL8_HF_VISIBILITY" \
  --pool-receipt "$FULL8_STAGE_ROOT/inventory/pool_corpus_receipt.json" \
  --packed-receipt "$FULL8_STAGE_ROOT/inventory/packed_corpus_receipt.json" \
  --packed-payload-integrity "$FULL8_PACKED_PAYLOAD_INTEGRITY" \
  --schedule-manifest "$FULL8_STAGE_ROOT/schedules/schedule_manifest.json" \
  --validation-manifest "$FULL8_STAGE_ROOT/validation/validation_manifest.json" \
  --token-bytes-receipt "$FULL8_STAGE_ROOT/validation/token_utf8_byte_lengths.receipt.json" \
  --benchmark-receipt "$FULL8_BENCHMARK_RECEIPT" \
  --graceful-stop-smoke "$FULL8_GRACEFUL_STOP_SMOKE" \
  --initial-validation "$FULL8_INITIAL_VALIDATION_RECEIPT" \
  --initial-greekmmlu "$FULL8_INITIAL_GREEKMMLU_RECEIPT" \
  --initial-per-document-root "$FULL8_INITIAL_PER_DOCUMENT_ROOT" \
  --initial-megatron "$FULL8_INITIAL_MEGATRON" \
  --initial-checkpoint-receipt "$FULL8_INITIAL_CHECKPOINT_RECEIPT" \
  --initial-hf-receipt "$FULL8_INITIAL_HF_RECEIPT" \
  --conversion-smoke "$FULL8_CONVERSION_SMOKE_RECEIPT" \
  --launch-environment "$environment" \
  --nested-sbatch-proof "$FULL8_NESTED_SBATCH_PROOF" \
  --output "$gate"

common=(
  FULL8_CODE_ROOT="$FULL8_CODE_ROOT"
  FULL8_CODE_BUNDLE_RECEIPT="$FULL8_CODE_BUNDLE_RECEIPT"
  FULL8_OPS_ROOT="$FULL8_OPS_ROOT"
  FULL8_OPS_BUNDLE_RECEIPT="$FULL8_OPS_BUNDLE_RECEIPT"
  FULL8_STAGE_ROOT="$FULL8_STAGE_ROOT"
  FULL8_RUN_ROOT="$FULL8_RUN_ROOT"
  FULL8_INITIAL_MEGATRON="$FULL8_INITIAL_MEGATRON"
  FULL8_PRELAUNCH_ROOT="$FULL8_PRELAUNCH_ROOT"
  FULL8_SELECTED_PROFILE="$FULL8_SELECTED_PROFILE"
  FULL8_LAUNCH_GATE="$gate"
  FULL8_OPERATIONAL_LAUNCH_GATE="$ops_gate"
  FULL8_RECIPE="$recipe"
  FULL8_PROFILES="$profiles"
  FULL8_TRAIN_LEAF_SWITCH="$FULL8_TRAIN_LEAF_SWITCH"
)
env "${common[@]}" DRY_RUN=1 \
  "$FULL8_OPS_ROOT/clariden/submit_production_resource_aware.sh" \
  >"$FULL8_PRELAUNCH_ROOT/production_submission_dry_run.out" \
  2>"$FULL8_PRELAUNCH_ROOT/production_submission_dry_run.err"

env "${common[@]}" DRY_RUN=0 CONFIRM_GPU_LAUNCH=APERTUS8B_FULL_MIXED_CPT \
  "$FULL8_OPS_ROOT/clariden/submit_production_resource_aware.sh" \
  | tee "$FULL8_PRELAUNCH_ROOT/production_submission.json"

if ! /usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/audit_submitted_job_resources.py" \
  --manifest "$graph" \
  --scientific-root "$FULL8_CODE_ROOT" \
  --scientific-receipt "$FULL8_CODE_BUNDLE_RECEIPT" \
  --ops-root "$FULL8_OPS_ROOT" \
  --ops-receipt "$FULL8_OPS_BUNDLE_RECEIPT" \
  --output "$routing"; then
  mapfile -t jobs < <(/usr/bin/python3.11 - "$graph" <<'PY'
import json,sys
print("\n".join(str(row["job_id"]) for row in json.load(open(sys.argv[1]))["jobs"]))
PY
)
  ((${#jobs[@]} == 0)) || scancel "${jobs[@]}"
  echo "resource audit failed; submitted jobs were cancelled" >&2
  exit 2
fi
