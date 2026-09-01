#!/usr/bin/env bash
#SBATCH --job-name=lr13_supp_bins
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00

set -euo pipefail
source "${LR13_CODE_ROOT:?set LR13_CODE_ROOT}/clariden/paths.env"
runtime="${LR13_BRISTEN_RUNTIME:-/iopsstor/scratch/cscs/fffoivos/python_envs/lr13_bristen_py311}"
supplement_root="$LR13_DATA_ROOT/replay_supplements"
builder="/iopsstor/scratch/cscs/fffoivos/repo/train-apertus-with-glossapi-cpt25b-86c1b8fe/subprojects/05_token_distillation_cpt/05_training_dataset_bridge/scripts/build_binary_shard.py"
"$runtime/bin/python" "$builder" \
  --input-receipt "$supplement_root/input_receipt.json" \
  --heldout-manifest "$supplement_root/heldouts/heldout_manifest.json" \
  --stage-root "$supplement_root" \
  --task-index "${SLURM_ARRAY_TASK_ID:?array task required}" \
  --workers 32 --chunksize 8
