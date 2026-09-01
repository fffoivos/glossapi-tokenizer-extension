#!/usr/bin/env bash
#SBATCH --job-name=neutral_source
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail

: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${NEUTRAL_RUN_ROOT:?set new neutral pipeline root}"
SOURCE_RUNTIME_PYTHON=${SOURCE_RUNTIME_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/cpt_build_xfer_py312/bin/python}
output="$NEUTRAL_RUN_ROOT/source"
[[ ! -e "$output" ]] || { echo "refusing to replace $output" >&2; exit 2; }
"$SOURCE_RUNTIME_PYTHON" \
  "$SCIENTIFIC_BUNDLE/evaluation/prepare_greek_parliament_neutral_source.py" \
  --output-root "$output" --reserve-clusters 512
