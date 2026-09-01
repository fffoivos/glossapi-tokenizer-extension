#!/usr/bin/env bash
#SBATCH --job-name=full8b-routing-proof
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable scientific code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set scientific bundle receipt}"
: "${FULL8_OPS_ROOT:?set immutable operational root}"
: "${FULL8_OPS_BUNDLE_RECEIPT:?set operational bundle receipt}"
: "${FULL8_ROUTING_SMOKE_ROOT:?set unique routing-smoke root}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "routing parent must be one debug node" >&2
  exit 2
}
[[ ! -e "$FULL8_ROUTING_SMOKE_ROOT" ]] || {
  echo "routing-smoke root already exists" >&2
  exit 2
}
mkdir -p "$FULL8_ROUTING_SMOKE_ROOT/logs"

/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_OPS_ROOT" --receipt "$FULL8_OPS_BUNDLE_RECEIPT" --kind efficiency

exports="ALL,FULL8_CODE_ROOT=$FULL8_CODE_ROOT,FULL8_CODE_BUNDLE_RECEIPT=$FULL8_CODE_BUNDLE_RECEIPT,FULL8_OPS_ROOT=$FULL8_OPS_ROOT,FULL8_OPS_BUNDLE_RECEIPT=$FULL8_OPS_BUNDLE_RECEIPT,FULL8_ROUTING_SMOKE_ROOT=$FULL8_ROUTING_SMOKE_ROOT,FULL8_ROUTING_PARENT_JOB=$SLURM_JOB_ID"
child=$(sbatch --uenv-passthrough=ignore --parsable \
  --partition=debug --time=00:03:00 --nodes=1 \
  --dependency="afterok:$SLURM_JOB_ID" \
  --job-name=full8b-routing-child \
  --output="$FULL8_ROUTING_SMOKE_ROOT/logs/%x-%j.out" \
  --error="$FULL8_ROUTING_SMOKE_ROOT/logs/%x-%j.err" \
  --export="$exports" \
  "$FULL8_OPS_ROOT/clariden/resource_aware_routing_child.sbatch")
child=${child%%;*}

/usr/bin/python3.11 - "$FULL8_ROUTING_SMOKE_ROOT/submission.json" "$SLURM_JOB_ID" "$child" <<'PY'
import datetime,json,os,sys,tempfile
out,parent,child=sys.argv[1:]
value={
 "schema_version":"apertus_full_8b_resource_routing_submission_v1",
 "status":"submitted",
 "submitted_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "parent_job_id":parent,"parent_partition":"debug",
 "child_job_id":child,"child_partition":"debug",
 "dependency":f"afterok:{parent}",
 "nested_submit_flag":"--uenv-passthrough=ignore",
}
fd,tmp=tempfile.mkstemp(prefix=".routing.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
printf '%s\n' "$child"
