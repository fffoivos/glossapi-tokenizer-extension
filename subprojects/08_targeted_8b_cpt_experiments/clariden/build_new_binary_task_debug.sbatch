#!/usr/bin/env bash
#SBATCH --job-name=t8b-build-bin
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=350G
#SBATCH --time=01:30:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
: "${TARGET8_BINARY_TASK_INDEX:?set exact new-modern task index}"
DATA_PYTHON=${TARGET8_DATA_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312/bin/python}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "new binary task must run on debug" >&2; exit 2; }
/usr/bin/python3.11 "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
contract="$TARGET8_STAGE_ROOT/experiment_a/binary_contract"
uenv run pytorch/v2.9.1:v2 --view=default -- "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts/build_binary_shard.py" \
  --input-receipt "$contract/input_receipt.json" --heldout-manifest "$contract/heldout_manifest.json" \
  --stage-root "$TARGET8_STAGE_ROOT/experiment_a" --task-index "$TARGET8_BINARY_TASK_INDEX" \
  --workers "${TARGET8_BINARY_WORKERS:-48}" --chunksize "${TARGET8_BINARY_CHUNKSIZE:-8}"
