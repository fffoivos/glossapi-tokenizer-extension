#!/usr/bin/env bash
set -euo pipefail
# /mnt/data starts as root-owned (or non-existent). Take ownership before anything else.
sudo mkdir -p /mnt/data
sudo chown -R "$(whoami):$(whoami)" /mnt/data
mkdir -p /mnt/data/run_state
chmod +x /home/foivos/worker/*.sh /home/foivos/worker/*.py 2>/dev/null || true

# Read HF_TOKEN from instance metadata server (only reachable from inside the VM).
HF_TOKEN=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token-secret")
WORKER_IDX=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-idx")
BUCKET=$(curl -fsS -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket")

export HF_TOKEN WORKER_IDX BUCKET

# Stage 1: bootstrap.
bash /home/foivos/worker/bootstrap.sh 2>&1 | tee -a /mnt/data/run_state/_run.log

source /mnt/data/profile.sh

# Stage 2: pull assigned shards.
cp /home/foivos/worker/worker_config.json /mnt/data/run_state/worker_config.json
python3 /home/foivos/worker/pull_assigned_shards.py 2>&1 | tee -a /mnt/data/run_state/_run.log

# Stage 3: hash pass.
cp /home/foivos/worker/text_dedup.py /mnt/data/text_dedup_lib.py
python3 /home/foivos/worker/hash_pass.py 2>&1 | tee -a /mnt/data/run_state/_run.log

# Stage 4: upload.
bash /home/foivos/worker/upload_output.sh 2>&1 | tee -a /mnt/data/run_state/_run.log

# Stage 5: done sentinel.
bash /home/foivos/worker/_done_sentinel.sh 2>&1 | tee -a /mnt/data/run_state/_run.log
