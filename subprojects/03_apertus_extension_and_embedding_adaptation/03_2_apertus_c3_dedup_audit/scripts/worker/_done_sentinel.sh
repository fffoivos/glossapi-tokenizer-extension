#!/usr/bin/env bash
# Final worker step: write the all-stages-done sentinel + summary line.
set -euo pipefail
source /mnt/data/profile.sh

# Verify all expected stage sentinels exist.
for f in bootstrap_done pull_done hash_done upload_done; do
  if [ ! -f "/mnt/data/$f" ]; then
    echo "MISSING: /mnt/data/$f" >&2
    exit 2
  fi
done

# Write the global sentinel.
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > /mnt/data/output/_done

# Upload the sentinel so the coordinator can detect it via the bucket too.
gcloud storage cp /mnt/data/output/_done "$BUCKET/worker_${WORKER_IDX}/_done" 2>&1 | head -3

echo "WORKER_DONE worker_${WORKER_IDX}"
