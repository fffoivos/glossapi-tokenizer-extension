#!/usr/bin/env bash
# Respawn ONE shard after a missing _DONE marker.
# Same instance config + metadata as the original; worker is idempotent
# (overwrites partial + _DONE on GCS).
#
# Usage:
#   SHARD=3 K=8 TIMESTAMP=20260518t012345 ZONE=europe-west4-a MACHINE=c4-highcpu-32 \
#       bash respawn_shard.sh
#
# Optional knobs:
#   BUCKET=gs://testbucketglossapi
#   HF_SECRET=hf-token-firing-count
#   SUFFIX=-r1   (appended to instance name to distinguish from previous attempt)

set -euo pipefail

: "${SHARD:?usage: SHARD=N K=K TIMESTAMP=... bash respawn_shard.sh}"
: "${K:?usage: SHARD=N K=K TIMESTAMP=... bash respawn_shard.sh}"
: "${TIMESTAMP:?usage: SHARD=N K=K TIMESTAMP=... bash respawn_shard.sh}"
ZONE=${ZONE:-europe-west4-a}
MACHINE=${MACHINE:-c4-highcpu-32}
BUCKET=${BUCKET:-gs://testbucketglossapi}
HF_SECRET=${HF_SECRET:-hf-token-firing-count}
SUFFIX=${SUFFIX:--r$(date -u +%H%M%S)}

SSP=$(cd "$(dirname "$0")/.." && pwd)
GCS_RUN_PREFIX="$BUCKET/firing_counts_$TIMESTAMP"
NAME="firing-w-$SHARD-$TIMESTAMP$SUFFIX"

echo "respawning shard $SHARD as $NAME"

gcloud compute instances create "$NAME" \
    --zone "$ZONE" \
    --machine-type "$MACHINE" \
    --boot-disk-size 200GB \
    --boot-disk-type hyperdisk-balanced \
    --image-family debian-12 --image-project debian-cloud \
    --scopes cloud-platform \
    --metadata "shard=$SHARD,total=$K,gcs-prefix=$GCS_RUN_PREFIX,hf-secret=$HF_SECRET" \
    --metadata-from-file "startup-script=$SSP/scripts/worker_startup.sh" \
    --labels "owner=foivos,purpose=firing-count,run=$TIMESTAMP,shard=$SHARD,respawn=true" \
    --quiet

TAG=$(printf 'shard_%02d_of_%02d' "$SHARD" "$K")
echo "  instance:  $NAME"
echo "  expecting: $GCS_RUN_PREFIX/_DONE_$TAG"
echo "  poll with: gcloud storage ls $GCS_RUN_PREFIX/_DONE_$TAG"
echo "  on fail:   gcloud compute ssh $NAME --zone $ZONE  # log at /var/log/firing_worker.log"
