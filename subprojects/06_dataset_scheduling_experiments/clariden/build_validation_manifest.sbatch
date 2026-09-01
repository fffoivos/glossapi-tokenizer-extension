#!/usr/bin/env bash
#SBATCH --job-name=mini_val_manifest
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
set -euo pipefail

: "${SCIENTIFIC_BUNDLE:?set immutable scientific bundle}"
: "${POOL_CORPUS_RECEIPT:?set frozen training pool receipt}"
: "${NEUTRAL_MANIFEST:?set completed neutral heldout manifest}"
: "${VALIDATION_MANIFEST:?set new output path}"
UENV_IMAGE=${UENV_IMAGE:-pytorch/v2.9.1:v2}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/greek_cpt25b_v1_py312}
[[ ! -e "$VALIDATION_MANIFEST" ]] || { echo "refusing to replace $VALIDATION_MANIFEST" >&2; exit 2; }
mkdir -p "$(dirname "$VALIDATION_MANIFEST")"
uenv run "$UENV_IMAGE" --view=default -- "$RUNTIME_VENV/bin/python" \
  "$SCIENTIFIC_BUNDLE/evaluation/build_validation_manifest.py" \
  --pool-corpus-receipt "$POOL_CORPUS_RECEIPT" \
  --neutral-manifest "$NEUTRAL_MANIFEST" --output "$VALIDATION_MANIFEST"
