#!/usr/bin/env bash
# Worker startup script — runs as root on instance boot, via instance
# metadata `startup-script`. Self-fetches its work from GCS, runs
# firing_count_worker.py, self-deletes on success.
#
# Instance metadata read:
#   shard          — shard index (0..K-1)
#   total          — total K
#   gcs-prefix     — gs://bucket/firing_counts_TS  (run root)
#   smoke          — "true" (smoke worker, --smoke flag + smoke output namespace)
#   smoke-max-rows — int, used only when smoke=true
#
# (No HF token needed: the tokenizer is staged as a local tar.gz on GCS
# by the coordinator; workers do not call HF Hub at runtime.)
#
# Inputs fetched from GCS:
#   gs://.../tokenizer.tar.gz        — chosen tokenizer (c3_added_17408_curated_padded)
#   gs://.../worker.py               — firing_count_worker.py
#   gs://.../shards/shard_NN_of_KK_text.parquet
#   gs://.../shards/shard_NN_of_KK_manifest.parquet
#
# Output: see firing_count_worker.py docstring (per-source counts +
# denoms + _DONE marker uploaded by the worker itself).
#
# Failure semantics: any error → log + exit non-zero → instance is NOT
# self-deleted. Coordinator's _DONE poll detects the missing marker and
# can SSH in for forensics (logs in /var/log/firing_worker.log).
#
# Cost containment: `timeout 30m` on the python entry point prevents a
# stuck worker from burning past the planned wall.

set -euo pipefail
exec > >(tee -a /var/log/firing_worker.log) 2>&1
echo "[$(date -Iseconds)] worker startup begins on $(hostname)"

# --- metadata helpers -------------------------------------------------------

meta() {
    curl -fsS "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
        -H 'Metadata-Flavor: Google'
}

meta_or() {
    local key="$1" default="$2"
    if curl -fsS "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$key" \
            -H 'Metadata-Flavor: Google' 2>/dev/null; then
        return 0
    fi
    echo "$default"
}

# --- instance identity ------------------------------------------------------

ZONE_URL=$(curl -fsS http://metadata.google.internal/computeMetadata/v1/instance/zone \
    -H 'Metadata-Flavor: Google')
ZONE=$(basename "$ZONE_URL")
INSTANCE=$(hostname)
echo "  zone=$ZONE  instance=$INSTANCE"

# --- read work metadata -----------------------------------------------------

SHARD=$(meta shard)
TOTAL=$(meta total)
GCS_PREFIX=$(meta gcs-prefix)
SMOKE=$(meta_or smoke "false")
SMOKE_MAX_ROWS=$(meta_or smoke-max-rows "10000")
echo "  shard=$SHARD total=$TOTAL smoke=$SMOKE"
echo "  gcs-prefix=$GCS_PREFIX"

# --- deps -------------------------------------------------------------------

echo "[$(date -Iseconds)] installing deps"
apt-get update -qq
apt-get install -y python3-pip python3-venv >/dev/null
python3 -m venv /opt/venv
/opt/venv/bin/pip install --quiet --upgrade pip
/opt/venv/bin/pip install --quiet \
    tokenizers pyarrow numpy psutil

# --- fetch inputs from GCS + Secret Manager --------------------------------

mkdir -p /work && cd /work

echo "[$(date -Iseconds)] fetching tokenizer + worker + shard data"
SHARD_TAG=$(printf 'shard_%02d_of_%02d' "$SHARD" "$TOTAL")
gcloud storage cp "$GCS_PREFIX/tokenizer.tar.gz" /work/tokenizer.tar.gz
gcloud storage cp "$GCS_PREFIX/worker.py"        /work/worker.py
gcloud storage cp "$GCS_PREFIX/shards/${SHARD_TAG}_text.parquet"     /work/text.parquet
gcloud storage cp "$GCS_PREFIX/shards/${SHARD_TAG}_manifest.parquet" /work/manifest.parquet
tar -xzf /work/tokenizer.tar.gz -C /work/   # → /work/c3_added_17408_curated_padded/

# --- run the worker (30-min cap) -------------------------------------------

EXTRA_ARGS=()
if [ "$SMOKE" == "true" ]; then
    EXTRA_ARGS+=(--smoke --max-total-rows-per-component "$SMOKE_MAX_ROWS")
fi

echo "[$(date -Iseconds)] launching worker (30m cap)"
set +e
timeout 30m /opt/venv/bin/python /work/worker.py \
    --text-parquet /work/text.parquet \
    --manifest     /work/manifest.parquet \
    --tokenizer-dir /work/c3_added_17408_curated_padded \
    --shard "$SHARD" --total "$TOTAL" \
    --gcs-out-prefix "$GCS_PREFIX" \
    --batch-size 4096 \
    --local-tmp /work \
    "${EXTRA_ARGS[@]}"
RC=$?
set -e
echo "[$(date -Iseconds)] worker exit code: $RC"

if [ "$RC" -ne 0 ]; then
    echo "[$(date -Iseconds)] WORKER FAILED — instance kept alive for forensics"
    echo "  inspect via: gcloud compute ssh $INSTANCE --zone $ZONE"
    echo "  log:         /var/log/firing_worker.log"
    exit "$RC"
fi

# --- success → self-delete --------------------------------------------------

echo "[$(date -Iseconds)] worker SUCCEEDED — self-deleting"
gcloud compute instances delete "$INSTANCE" --zone "$ZONE" --quiet || {
    echo "self-delete failed; instance will linger. Manually delete if not investigating."
}
