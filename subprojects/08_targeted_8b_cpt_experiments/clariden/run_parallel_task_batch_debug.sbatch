#!/usr/bin/env bash
#SBATCH --job-name=t8b-task-batch
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=450G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
: "${TARGET8_TASK_KIND:?set binary or packing}"
: "${TARGET8_TASK_START:?set inclusive positional task start}"
: "${TARGET8_TASK_END:?set exclusive positional task end}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
PARALLEL=${TARGET8_TASK_PARALLEL:-8}
WORKERS_PER_TASK=${TARGET8_WORKERS_PER_TASK:-16}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "task batches must run on debug" >&2; exit 2; }
[[ "$TARGET8_TASK_KIND" == binary || "$TARGET8_TASK_KIND" == packing ]] || { echo "invalid task kind" >&2; exit 2; }

/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
receipt="$TARGET8_STAGE_ROOT/experiment_a/task_batches/${TARGET8_TASK_KIND}_${TARGET8_TASK_START}_${TARGET8_TASK_END}.json"
mkdir -p "$(dirname "$receipt")"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/run_parallel_task_batch.py" \
  --kind "$TARGET8_TASK_KIND" --code-root "$TARGET8_CODE_ROOT" --stage-root "$TARGET8_STAGE_ROOT" \
  --data-python "$DATA_PYTHON" --start "$TARGET8_TASK_START" --end "$TARGET8_TASK_END" \
  --parallel "$PARALLEL" --workers-per-task "$WORKERS_PER_TASK" --receipt "$receipt"
