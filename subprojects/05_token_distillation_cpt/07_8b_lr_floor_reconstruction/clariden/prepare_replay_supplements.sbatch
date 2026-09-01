#!/usr/bin/env bash
#SBATCH --job-name=lr13_supp_plan
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

set -euo pipefail
source "${LR13_CODE_ROOT:?set LR13_CODE_ROOT}/clariden/paths.env"
runtime="${LR13_BRISTEN_RUNTIME:-/iopsstor/scratch/cscs/fffoivos/python_envs/lr13_bristen_py311}"
"$runtime/bin/python" "$LR13_CODE_ROOT/dataset/prepare_replay_supplements.py" \
  --recipe "$LR13_RECIPE" \
  --code-root "$LR13_CODE_ROOT" \
  --output-root "$LR13_DATA_ROOT/replay_supplements"
