#!/usr/bin/env bash
#SBATCH --job-name=mini-docval-manifest
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --no-requeue

set -euo pipefail
: "${FULL8_CODE_ROOT:?set immutable CSCS code root}"
: "${MINI_VALIDATION_MANIFEST:?set frozen Mini validation manifest}"
: "${MINI_DOCVAL_MANIFEST:?set new per-document manifest path}"
uenv run pytorch/v2.9.1:v2 --view=default -- python3 \
  "$FULL8_CODE_ROOT/subprojects/07_full_8b_cpt/evaluation/build_mini_per_document_manifest.py" \
  --mini-manifest "$MINI_VALIDATION_MANIFEST" --output "$MINI_DOCVAL_MANIFEST"
