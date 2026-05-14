#!/usr/bin/env bash
# Bootstrap a c4-highcpu-192 worker for the vocab-attribution run.
# Usage on worker: bash entrypoint.sh <WORKER_IDX>
# Requires: HF_TOKEN in env, GCS_BUCKET in env, /mnt/data on root.

set -euo pipefail

WORKER_IDX="${1:?usage: bash entrypoint.sh <WORKER_IDX>}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN not set" >&2; exit 2
fi
if [[ -z "${GCS_BUCKET:-}" ]]; then
  echo "FATAL: GCS_BUCKET not set" >&2; exit 2
fi
if [[ -z "${RUN_ID:-}" ]]; then
  echo "FATAL: RUN_ID not set" >&2; exit 2
fi

echo "[entrypoint] worker $WORKER_IDX, run $RUN_ID, bucket $GCS_BUCKET"

echo "[entrypoint] apt deps"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3 python3-pip python3-venv git tmux htop nvme-cli jq curl

echo "[entrypoint] preparing /mnt/data on root disk"
sudo growpart /dev/nvme0n1 1 || true
sudo resize2fs /dev/nvme0n1p1 || true
sudo mkdir -p /mnt/data
sudo chown -R "${USER}:${USER}" /mnt/data
mkdir -p /mnt/data/{scratch,outputs,logs,hf_cache}

echo "[entrypoint] venv"
python3 -m venv /mnt/data/venv
source /mnt/data/venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet \
  "tokenizers>=0.20" \
  "transformers>=4.45" \
  "huggingface_hub>=0.25" \
  "pyarrow>=17" \
  "datasets>=3.0" \
  "numpy>=2.0"

echo "[entrypoint] env profile"
cat > /mnt/data/profile.sh <<EOF
source /mnt/data/venv/bin/activate
export HF_TOKEN='${HF_TOKEN}'
export HF_HUB_TOKEN='${HF_TOKEN}'
export HUGGINGFACE_HUB_CACHE=/mnt/data/hf_cache
export HF_DATASETS_CACHE=/mnt/data/hf_cache
export TRANSFORMERS_CACHE=/mnt/data/hf_cache
export RAYON_NUM_THREADS=192
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export GCS_BUCKET='${GCS_BUCKET}'
export RUN_ID='${RUN_ID}'
export WORKER_IDX='${WORKER_IDX}'
EOF
chmod +x /mnt/data/profile.sh
source /mnt/data/profile.sh

echo "[entrypoint] tokenizer prefetch"
python3 - <<'PY'
from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")
print("tokenizer vocab:", tok.get_vocab_size())
PY

echo "[entrypoint] ready. worker_idx=${WORKER_IDX}"
