#!/usr/bin/env bash
#SBATCH --job-name=t8b-build-a-sched
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "schedule construction must run on debug" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
stage="$TARGET8_STAGE_ROOT/experiment_a"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/dataset/build_five_schedules.py" \
  --packed-receipt "$stage/packed_corpus_receipt.json" --output-dir "$stage/schedules" \
  --global-batch-sequences 1024 --arm D0_mixed
