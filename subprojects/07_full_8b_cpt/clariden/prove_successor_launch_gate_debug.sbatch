#!/usr/bin/env bash
#SBATCH --job-name=full8b-successor-gate-debug
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

# Execute the complete successor launch gate without submitting any normal job.
set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_STAGE_ROOT:?set successor data root}"
: "${FULL8_RECIPE:?set versioned successor recipe}"
: "${FULL8_PROFILES:?set versioned successor profiles}"
: "${FULL8_SELECTED_PROFILE:?set successor selected profile}"
: "${FULL8_SOURCE_SELECTED_PROFILE:?set proven source selected profile}"
: "${FULL8_BENCHMARK_RECEIPT:?set proven DP32 promotion receipt}"
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
: "${FULL8_HF_VISIBILITY:?set public HF visibility receipt}"
: "${FULL8_RESOURCE_ROUTING_SMOKE:?set v7 resource-routing smoke receipt}"
: "${FULL8_EVAL_OVERLAP_SMOKE:?set v7 evaluation-overlap smoke receipt}"
: "${FULL8_TRAIN_LEAF_SWITCH:?set pinned Clariden leaf switch}"
: "${FULL8_GATE_PROOF_ROOT:?set unique debug-only proof root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "successor gate proof must run on one debug node" >&2
  exit 2
}
[[ ! -e "$FULL8_GATE_PROOF_ROOT" ]] || { echo "gate-proof root exists" >&2; exit 2; }
mkdir -p "$FULL8_GATE_PROOF_ROOT"

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

/usr/bin/python3.11 - "$FULL8_RESOURCE_ROUTING_SMOKE" "$FULL8_EVAL_OVERLAP_SMOKE" \
  "$FULL8_CODE_ROOT" "$FULL8_CODE_BUNDLE_RECEIPT" "$FULL8_OPS_ROOT" "$FULL8_OPS_BUNDLE_RECEIPT" <<'PY'
import json,os,sys
routing,overlap,code_root,code_receipt,ops_root,ops_receipt=sys.argv[1:]
def read(path): return json.load(open(path))
r=read(routing); e=read(overlap)
checks=(
 r.get("schema_version")=="apertus_full_8b_resource_routing_smoke_v1",
 r.get("status")=="passed", all(r.get("checks",{}).values()),
 r.get("parent_partition")==r.get("child_partition")=="debug", r.get("child_nodes")==1,
 r.get("nested_submit_flag")=="--uenv-passthrough=ignore",
 e.get("schema_version")=="apertus_full_8b_evaluation_overlap_smoke_v1",
 e.get("status")=="passed", all(e.get("checks",{}).values()),
 e.get("partition")=="debug" and e.get("nodes")==4,
 len(e.get("steps",[]))==5 and all(row.get("exit_code")==0 for row in e["steps"]),
 os.path.realpath(r.get("scientific_root",""))==os.path.realpath(code_root),
 os.path.realpath(e.get("scientific_root",""))==os.path.realpath(code_root),
 os.path.realpath(r.get("scientific_receipt",""))==os.path.realpath(code_receipt),
 os.path.realpath(e.get("scientific_receipt",""))==os.path.realpath(code_receipt),
 os.path.realpath(r.get("operational_root",""))==os.path.realpath(ops_root),
 os.path.realpath(e.get("operational_root",""))==os.path.realpath(ops_root),
 os.path.realpath(r.get("operational_receipt",""))==os.path.realpath(ops_receipt),
 os.path.realpath(e.get("operational_receipt",""))==os.path.realpath(ops_receipt),
)
if not all(checks): raise SystemExit("operational smoke evidence drift")
PY

environment="$FULL8_GATE_PROOF_ROOT/launch_environment.json"
gate="$FULL8_GATE_PROOF_ROOT/launch_gate.json"
train_exclude=$("$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/clariden/resolve_leaf_switch_exclusion.sh" "$FULL8_TRAIN_LEAF_SWITCH" 16)
/usr/bin/python3.11 "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/scripts/capture_launch_environment.py" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --leaf-switch "$FULL8_TRAIN_LEAF_SWITCH" --leaf-exclusion "$train_exclude" \
  --storage-path /capstor/scratch/cscs/fffoivos/runs --output "$environment"

owner=${FULL8_OWNER_DECISIONS:-$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/configs/owner_decisions_20260805.json}
/usr/bin/python3.11 "$FULL8_OPS_ROOT/scripts/build_successor_launch_gate.py" \
  --code-bundle-receipt "$FULL8_CODE_BUNDLE_RECEIPT" --code-root "$FULL8_CODE_ROOT" \
  --recipe "$FULL8_RECIPE" --profiles "$FULL8_PROFILES" \
  --selected-profile "$FULL8_SELECTED_PROFILE" \
  --source-selected-profile "$FULL8_SOURCE_SELECTED_PROFILE" \
  --source-promotion-receipt "$FULL8_BENCHMARK_RECEIPT" \
  --stage-identity "$FULL8_STAGE_ROOT/inventory/successor_stage_identity.json" \
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
  --launch-environment "$environment" --nested-sbatch-proof "$FULL8_NESTED_SBATCH_PROOF" \
  --output "$gate"

/usr/bin/python3.11 - "$FULL8_GATE_PROOF_ROOT/gate_proof_receipt.json" "$environment" "$gate" <<'PY'
import datetime,hashlib,json,os,sys,tempfile
out,environment,gate=sys.argv[1:]
def bind(path):
 raw=open(path,"rb").read()
 return {"path":os.path.realpath(path),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}
value={"schema_version":"apertus_full_8b_successor_launch_gate_proof_v1","status":"passed",
 "completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "policy":"debug_only_full_gate_proof_no_normal_submission_v1",
 "launch_environment":bind(environment),"launch_gate":bind(gate)}
fd,tmp=tempfile.mkstemp(prefix=".gate-proof.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
print(json.dumps({"ok":True,"gate":value["launch_gate"]["sha256"]},sort_keys=True))
PY
