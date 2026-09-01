#!/usr/bin/env bash
#SBATCH --job-name=full8-anon-binary
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=06:00:00
#SBATCH --no-requeue

set -euo pipefail
HOST_PYTHON=${HOST_PYTHON:-/usr/bin/python3.11}
DATA_PYTHON=${FULL8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
DATA_RUNTIME_RECEIPT=${FULL8_DATA_RUNTIME_RECEIPT:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/runtime_receipt.json}
: "${FULL8_CODE_ROOT:?set immutable code root}"
: "${FULL8_CODE_BUNDLE_RECEIPT:?set immutable code-bundle receipt}"
: "${FULL8_ANON_STAGE_ROOT:?set anonymized derivative root}"
: "${FULL8_ANON_TASK_COUNT:?set exact frozen task count}"
test -s "$DATA_RUNTIME_RECEIPT"
GROUP_SIZE=${FULL8_ANON_GROUP_SIZE:-4}
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "array task required" >&2; exit 2; }
"$HOST_PYTHON" "$FULL8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$FULL8_CODE_ROOT" --receipt "$FULL8_CODE_BUNDLE_RECEIPT" --kind scientific
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" -c \
  'import pyarrow,tokenizers; assert pyarrow.__version__ == "21.0.0"'

start=$((SLURM_ARRAY_TASK_ID * GROUP_SIZE))
pids=()
for ((offset=0; offset<GROUP_SIZE; offset++)); do
  index=$((start + offset))
  ((index < FULL8_ANON_TASK_COUNT)) || break
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=72 --mem=110G \
    uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
    "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/dataset/anonymization/build_sanitized_binary_shard.py" \
    --overlay "$FULL8_ANON_STAGE_ROOT/anonymization_overlay.json" \
    --heldout-manifest "$FULL8_ANON_STAGE_ROOT/heldouts/heldout_manifest.json" \
    --dedup-receipt "$FULL8_ANON_STAGE_ROOT/postmask_dedup/dedup_receipt.json" \
    --stage-root "$FULL8_ANON_STAGE_ROOT" --task-index "$index" --workers 64 --chunksize 8 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
