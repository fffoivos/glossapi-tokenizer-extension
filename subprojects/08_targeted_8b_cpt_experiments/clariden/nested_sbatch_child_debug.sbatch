#!/usr/bin/env bash
#SBATCH --job-name=t8b-nested-submit-child
#SBATCH --account=a0140
#SBATCH --partition=debug
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
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || {
  echo "nested-submit child must run on debug" >&2
  exit 2
}
MEGATRON=${FULL8_MEGATRON_ROOT:-/iopsstor/scratch/cscs/fffoivos/orchestration/dataset-scheduling-0p5b/20260803T093500Z-megatron-production-c92402e-v1}
TRAINING_CODE="$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/training"
RUNTIME_COMPAT="$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/runtime_compat"
/usr/bin/python3.11 \
  "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific

rank_runtime=$(srun --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=1 --cpu-bind=cores \
  bash -lc "exec uenv run pytorch/v2.9.1:v2 --view=default -- bash -lc 'export PYTHONPATH=\"$RUNTIME_COMPAT:$MEGATRON:$TRAINING_CODE\"; command -v torchrun; python3 -c \"import megatron\"; printf \"%s\\n\" megatron-import-ok'")
rank_torchrun=$(printf '%s\n' "$rank_runtime" | sed -n '1p')
rank_megatron=$(printf '%s\n' "$rank_runtime" | sed -n '2p')
[[ -n "$rank_torchrun" && "$rank_megatron" == megatron-import-ok ]] || {
  echo "rank-local uenv did not expose torchrun and import megatron" >&2
  exit 2
}

export FULL8_NESTED_RANK_TORCHRUN=$rank_torchrun
export FULL8_NESTED_RANK_MEGATRON=$rank_megatron
/usr/bin/python3.11 - "$FULL8_NESTED_SMOKE_ROOT/nested_sbatch_proof.json" <<'PY'
import datetime
import json
import os
import sys
import tempfile

output = sys.argv[1]
value = {
    "schema_version": "apertus_full_8b_nested_sbatch_proof_v1",
    "status": "passed",
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "parent_job_id": os.environ["FULL8_NESTED_PARENT_JOB_ID"],
    "child_job_id": os.environ["SLURM_JOB_ID"],
    "code_root": os.path.realpath(os.environ["FULL8_CODE_ROOT"]),
    "code_bundle_receipt": os.path.realpath(os.environ["FULL8_CODE_BUNDLE_RECEIPT"]),
    "parent_runtime": "uenv run pytorch/v2.9.1:v2 --view=default -- python3",
    "nested_submit_flag": "--uenv-passthrough=ignore",
    "rank_runtime": "uenv run pytorch/v2.9.1:v2 --view=default -- torchrun",
    "rank_torchrun": os.environ["FULL8_NESTED_RANK_TORCHRUN"],
    "rank_megatron_import": os.environ["FULL8_NESTED_RANK_MEGATRON"],
}
fd, temporary = tempfile.mkstemp(
    prefix=".nested_sbatch_proof.", suffix=".partial", dir=os.path.dirname(output)
)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(temporary, output)
PY
