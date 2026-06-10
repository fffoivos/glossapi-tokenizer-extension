#!/usr/bin/env bash
#SBATCH --job-name=cpt13b_xfer_env
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.err
# Build an x86_64 Python env for CPU-only xfer corpus jobs.
set -euo pipefail

SC=/iopsstor/scratch/cscs/fffoivos
UV=/users/fffoivos/.local/bin/uv
PY312=/users/fffoivos/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
VENV=$SC/python_envs/cpt_build_xfer_py312

test -x "$UV"
test -x "$PY312"
mkdir -p "$(dirname "$VENV")"

if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv --python "$PY312" "$VENV"
fi
"$UV" pip install --python "$VENV/bin/python" \
  datasets==5.0.0 \
  datatrove==0.8.0 \
  pyarrow==24.0.0 \
  tokenizers==0.22.2 \
  transformers==5.10.2 \
  numpy==2.4.1 \
  einops \
  sentencepiece \
  protobuf \
  orjson \
  regex \
  tqdm \
  safetensors
"$UV" pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.9.1

"$VENV/bin/python" - <<'PY'
import datatrove, datasets, numpy, pyarrow, tokenizers, torch, transformers
print("xfer env OK")
print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("transformers", transformers.__version__)
PY
