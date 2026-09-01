#!/usr/bin/env bash
#SBATCH --job-name=mini_base_freeze
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
: "${MINI_CODE_BUNDLE:?set immutable Mini code bundle}"
: "${MINI_BASE_MODEL_DIR:?set immutable output directory}"
BOOTSTRAP_PYTHON=${BOOTSTRAP_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/cpt_build_xfer_py312/bin/python}
RUNTIME_VENV=${RUNTIME_VENV:-/iopsstor/scratch/cscs/fffoivos/python_envs/mini_assets_xfer_py312}
mkdir -p "$(dirname "$RUNTIME_VENV")"
(
  flock 9
  if [[ ! -x "$RUNTIME_VENV/bin/python" ]]; then
    "$BOOTSTRAP_PYTHON" -m venv "$RUNTIME_VENV"
    "$RUNTIME_VENV/bin/python" -m pip install --upgrade pip
    "$RUNTIME_VENV/bin/python" -m pip install \
      'huggingface_hub==0.36.0' 'transformers==4.57.0' \
      'tokenizers==0.22.1' safetensors
  fi
) 9>"$RUNTIME_VENV.lock"
"$RUNTIME_VENV/bin/python" \
  "$MINI_CODE_BUNDLE/initialization/freeze_base_mini_model.py" \
  --output-dir "$MINI_BASE_MODEL_DIR" \
  --repo-id swiss-ai/Apertus-v1.1-0.5B \
  --revision 1b7276176e564fc0cc7d7c3b991a8d653c8b8792
