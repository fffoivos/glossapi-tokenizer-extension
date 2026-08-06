#!/usr/bin/env bash
#SBATCH --job-name=full8b-nested-submit-child
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code receipt}"
: "${FULL8_NESTED_SMOKE_ROOT:?set nested-submit smoke root}"
: "${FULL8_NESTED_PARENT_JOB_ID:?set parent Slurm job id}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
export FULL8_NESTED_RANK_TORCHRUN
FULL8_NESTED_RANK_TORCHRUN=$(uenv run pytorch/v2.9.1:v2 --view=default -- \
  srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 --cpu-bind=cores \
  bash -lc 'exec uenv run pytorch/v2.9.1:v2 --view=default -- bash -lc "command -v torchrun"')
[[ -n "$FULL8_NESTED_RANK_TORCHRUN" ]] || { echo "rank-local uenv did not expose torchrun" >&2; exit 2; }
python3 - "$FULL8_NESTED_SMOKE_ROOT/nested_sbatch_proof.json" <<'PY'
import datetime,json,os,sys,tempfile
out=sys.argv[1]
value={
 "schema_version":"apertus_full_8b_nested_sbatch_proof_v1",
 "status":"passed",
 "completed_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
 "parent_job_id":os.environ["FULL8_NESTED_PARENT_JOB_ID"],
 "child_job_id":os.environ["SLURM_JOB_ID"],
 "code_root":os.path.realpath(os.environ["FULL8_CODE_ROOT"]),
 "code_bundle_receipt":os.path.realpath(os.environ["FULL8_CODE_BUNDLE_RECEIPT"]),
 "parent_runtime":"uenv run pytorch/v2.9.1:v2 --view=default -- python3",
 "nested_submit_flag":"--uenv-passthrough=ignore",
 "rank_runtime":"uenv run pytorch/v2.9.1:v2 --view=default -- torchrun",
 "rank_torchrun":os.environ["FULL8_NESTED_RANK_TORCHRUN"],
}
fd,tmp=tempfile.mkstemp(prefix=".nested_sbatch_proof.",suffix=".partial",dir=os.path.dirname(out))
with os.fdopen(fd,"w") as f:
 json.dump(value,f,indent=2,sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
os.replace(tmp,out)
PY
