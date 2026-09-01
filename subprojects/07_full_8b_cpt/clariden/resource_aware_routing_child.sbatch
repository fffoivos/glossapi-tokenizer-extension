#!/usr/bin/env bash
#SBATCH --job-name=full8b-routing-child
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:03:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_ROUTING_SMOKE_ROOT:?set routing-smoke root}"
: "${FULL8_ROUTING_PARENT_JOB:?set parent job id}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "routing child must be one debug node" >&2
  exit 2
}

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

parent_state=$(sacct -X -n -P -j "$FULL8_ROUTING_PARENT_JOB" --format=State | head -1 | cut -d'|' -f1 | cut -d' ' -f1 | cut -d'+' -f1)
[[ "$parent_state" == COMPLETED ]] || {
  echo "routing parent was not complete before child start: $parent_state" >&2
  exit 2
}

/usr/bin/python3.11 - \
  "$FULL8_ROUTING_SMOKE_ROOT/resource_routing_smoke_receipt.json" \
  "$FULL8_ROUTING_PARENT_JOB" "$SLURM_JOB_ID" "$parent_state" \
  "$FULL8_CODE_ROOT" "$FULL8_CODE_BUNDLE_RECEIPT" \
  "$FULL8_OPS_ROOT" "$FULL8_OPS_BUNDLE_RECEIPT" <<'PY'
import datetime,json,os,sys,tempfile
out,parent,child,parent_state,code_root,code_receipt,ops_root,ops_receipt=sys.argv[1:]
checks={
 "parent_completed_before_child":parent_state=="COMPLETED",
 "child_partition_is_debug":os.environ.get("SLURM_JOB_PARTITION")=="debug",
 "child_nodes_are_one":os.environ.get("SLURM_NNODES")=="1",
 "scientific_bundle_exists":os.path.isdir(code_root) and os.path.isfile(code_receipt),
 "operational_bundle_exists":os.path.isdir(ops_root) and os.path.isfile(ops_receipt),
}
if not all(checks.values()): raise SystemExit(checks)
value={
 "schema_version":"apertus_full_8b_resource_routing_smoke_v1",
 "status":"passed","completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "parent_job_id":parent,"child_job_id":child,
 "parent_state":parent_state,"parent_partition":"debug",
 "child_partition":"debug","child_nodes":1,
 "nested_submit_flag":"--uenv-passthrough=ignore",
 "scientific_root":os.path.realpath(code_root),
 "scientific_receipt":os.path.realpath(code_receipt),
 "operational_root":os.path.realpath(ops_root),
 "operational_receipt":os.path.realpath(ops_receipt),
 "checks":checks,
}
fd,tmp=tempfile.mkstemp(prefix=".routing.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
print(json.dumps({"ok":True,"parent":parent,"child":child},sort_keys=True))
PY
