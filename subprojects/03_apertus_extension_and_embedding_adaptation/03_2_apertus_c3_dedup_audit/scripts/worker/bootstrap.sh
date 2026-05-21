#!/usr/bin/env bash
# Worker bootstrap — runs once when the worker boots.
# Inputs: $WORKER_IDX, $HF_TOKEN, $BUCKET (passed via metadata or env)
# Outputs: /mnt/data/{venv,sources,output} prepared; touches /mnt/data/bootstrap_done

set -euo pipefail

WORKER_IDX="${WORKER_IDX:?WORKER_IDX env required}"
# c4-highcpu-192 has no local SSDs in this config; use the 500GB hyperdisk-balanced boot disk
# (root fs is auto-resized to use it). Plenty of room for ~43GB per-worker shards + outputs.
sudo mkdir -p /mnt/data
sudo chown -R "$(whoami):$(whoami)" /mnt/data
LOG=/mnt/data/bootstrap.log
echo "=== worker $WORKER_IDX bootstrap start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
df -h / | tee -a "$LOG"

mkdir -p /mnt/data/{sources,output,logs,run_state}

# Install deps via apt + venv.
echo "Installing apt deps..." | tee -a "$LOG"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  python3-pip python3-venv python3-dev build-essential google-cloud-cli 2>&1 | tail -5 | tee -a "$LOG"

echo "Creating venv at /mnt/data/venv..." | tee -a "$LOG"
python3 -m venv /mnt/data/venv
source /mnt/data/venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet \
  pyarrow==17.0.0 polars==1.13.0 \
  huggingface_hub==0.27.0 \
  numpy==2.1.3 regex blake3 \
  unicodedata2 zstandard tqdm 2>&1 | tail -3 | tee -a "$LOG"

# Write profile.sh with run env.
cat >/mnt/data/profile.sh <<EOF
export RAYON_NUM_THREADS=192
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=192
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN env required}"
export HF_HOME=/mnt/data/hf_cache
export WORKER_IDX=${WORKER_IDX}
export BUCKET="${BUCKET:?BUCKET env required}"
source /mnt/data/venv/bin/activate
EOF
chmod +x /mnt/data/profile.sh
echo "profile.sh written" | tee -a "$LOG"

# Sanity: verify python deps + gsutil work.
. /mnt/data/profile.sh
python3 -c "import pyarrow, polars, blake3, huggingface_hub; print('deps OK')" | tee -a "$LOG"
gcloud --version | head -2 | tee -a "$LOG"

touch /mnt/data/bootstrap_done
echo "=== worker $WORKER_IDX bootstrap done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
