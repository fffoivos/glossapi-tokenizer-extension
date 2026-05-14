#!/usr/bin/env bash
# Bootstrap a fresh c4-highcpu-192 worker for the Greek-share Path-A run.
# Run as: bash entrypoint.sh
# Requires: HF_TOKEN exported in the caller environment, /mnt/data prepared as local-SSD scratch.

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "FATAL: HF_TOKEN not set" >&2
  exit 2
fi

echo "[entrypoint] installing system deps"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3 python3-pip python3-venv git tmux htop nvme-cli jq curl

echo "[entrypoint] preparing /mnt/data on root disk"
# This worker has hyperdisk-balanced as the boot disk (~2 TB), no local SSD.
# /mnt/data lives on the root filesystem; ensure root partition is grown.
sudo growpart /dev/nvme0n1 1 || true
sudo resize2fs /dev/nvme0n1p1 || true
sudo mkdir -p /mnt/data
sudo chown -R "${USER}:${USER}" /mnt/data
df -h / | tee /tmp/df.log

mkdir -p /mnt/data/scratch /mnt/data/outputs /mnt/data/logs /mnt/data/hf_cache

echo "[entrypoint] creating venv"
python3 -m venv /mnt/data/venv
# shellcheck disable=SC1091
source /mnt/data/venv/bin/activate

echo "[entrypoint] pip installing"
pip install --quiet --upgrade pip
pip install --quiet \
  "tokenizers>=0.20" \
  "transformers>=4.45" \
  "huggingface_hub>=0.25" \
  "pyarrow>=17" \
  "datasets>=3.0" \
  "polars>=1.10"

echo "[entrypoint] writing env profile"
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
export POLARS_MAX_THREADS=8
EOF
chmod +x /mnt/data/profile.sh

echo "[entrypoint] verifying tokenizer download"
source /mnt/data/profile.sh
python3 - <<'PY'
import os
from tokenizers import Tokenizer
tok = Tokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")
print("tokenizer vocab size:", tok.get_vocab_size())
PY

echo "[entrypoint] ready. Activate with: source /mnt/data/profile.sh"
