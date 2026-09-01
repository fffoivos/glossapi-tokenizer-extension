#!/usr/bin/env bash
#SBATCH --job-name=lr13_supp_download
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
source "${LR13_CODE_ROOT:?set LR13_CODE_ROOT}/clariden/paths.env"
runtime="${LR13_BRISTEN_RUNTIME:-/iopsstor/scratch/cscs/fffoivos/python_envs/lr13_bristen_py311}"
"$runtime/bin/python" "$LR13_CODE_ROOT/dataset/download_replay_supplements.py" \
  --recipe "$LR13_RECIPE" \
  --output-root "$LR13_DATA_ROOT/replay_supplements"
