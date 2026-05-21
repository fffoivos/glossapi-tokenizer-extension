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

failure_report() {
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    return 0
  fi
  mkdir -p /mnt/data/run_state
  cat >/mnt/data/run_state/_failed.json <<EOF
{"worker_idx": ${WORKER_IDX:-null}, "exit_code": $rc, "failed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
  # Best-effort upload. This is intentionally outside upload_output.sh because
  # failures before hash_done must still become visible to the coordinator.
  if command -v gcloud >/dev/null 2>&1 && [ -n "${BUCKET:-}" ] && [ -n "${WORKER_IDX:-}" ]; then
    PREFIX="$BUCKET/worker_${WORKER_IDX}"
    gcloud storage cp /mnt/data/run_state/_failed.json "$PREFIX/_failed" >/dev/null 2>&1 || true
    gcloud storage cp /mnt/data/run_state/*.jsonl "$PREFIX/run_state/" >/dev/null 2>&1 || true
    gcloud storage cp /mnt/data/run_state/_run.log "$PREFIX/logs/_run.log" >/dev/null 2>&1 || true
    [ -f /mnt/data/bootstrap.log ] && gcloud storage cp /mnt/data/bootstrap.log "$PREFIX/logs/bootstrap.log" >/dev/null 2>&1 || true
    [ -f /home/foivos/worker/run_all.out ] && gcloud storage cp /home/foivos/worker/run_all.out "$PREFIX/logs/run_all.out" >/dev/null 2>&1 || true
  fi
  exit "$rc"
}
trap failure_report EXIT

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
