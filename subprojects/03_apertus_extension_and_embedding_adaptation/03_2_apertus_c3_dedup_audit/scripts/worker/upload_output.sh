#!/usr/bin/env bash
# Worker step: upload /mnt/data/output/*.parquet + logs to the run's GCS bucket.
# Inputs: $BUCKET (from /mnt/data/profile.sh), $WORKER_IDX
set -euo pipefail
source /mnt/data/profile.sh

LOG=/mnt/data/upload.log
echo "=== worker $WORKER_IDX upload start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

PREFIX="$BUCKET/worker_${WORKER_IDX}"

# Upload all shard outputs in parallel via gcloud storage cp -r (parallelizes per file).
gcloud storage cp -r /mnt/data/output/*.parquet "$PREFIX/" 2>&1 | tee -a "$LOG"
gcloud storage cp /mnt/data/run_state/*.jsonl "$PREFIX/run_state/" 2>&1 | tee -a "$LOG"
gcloud storage cp /mnt/data/bootstrap.log /mnt/data/upload.log "$PREFIX/logs/" 2>&1 | tee -a "$LOG" || true

# Write a small manifest of what was uploaded.
ls /mnt/data/output/*.parquet | wc -l > /tmp/_n
N=$(cat /tmp/_n)
cat >/mnt/data/upload_manifest.json <<EOF
{"worker_idx": $WORKER_IDX, "shard_count": $N, "prefix": "$PREFIX", "uploaded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
gcloud storage cp /mnt/data/upload_manifest.json "$PREFIX/manifest.json" 2>&1 | tee -a "$LOG"

touch /mnt/data/upload_done
echo "=== worker $WORKER_IDX upload done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
