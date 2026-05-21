#!/usr/bin/env bash
# Coordinator step 03: scp scripts + per-worker config to each worker, attach
# HF_TOKEN via instance metadata (NOT baked into scripts), then launch the
# pipeline detached.
#
# Per review r4: HF_TOKEN must NOT be written into worker scripts in plaintext.
# Instead we set it as instance metadata key `hf-token-secret`; the worker
# reads it at boot time from the metadata server (only accessible from inside
# the VM).
set -euo pipefail

SUB="/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit"
RUN_ID="${RUN_ID:-$(cat "$SUB/manifests/CURRENT_RUN_ID")}"
RUN_LOWER="${RUN_ID,,}"
NAME_SUFFIX="${RUN_LOWER//_/-}"
MANI="$SUB/manifests/run_$RUN_ID"
REPO="/home/foivos/Projects/glossapi-tokenizer-extension"
# Zone is now read per-worker from workers.list (TSV: name<TAB>zone).

LOG="$MANI/dispatch.log"
echo "=== dispatch start $(date -u +%Y-%m-%dT%H:%M:%SZ) RUN_ID=$RUN_ID ===" | tee -a "$LOG"

PIN_JSON="$MANI/text_dedup_pin.json"
BUCKET=$(python3 -c "import json; print(json.load(open('$PIN_JSON'))['bucket'])")
HF_TOKEN_VAL="${HF_TOKEN:?HF_TOKEN env required}"

# Compose a per-worker run_all.sh that reads HF_TOKEN from instance metadata.
# This file is COMMITTED to the worker filesystem but contains no secrets.
cat >"$MANI/run_all_on_worker.sh" <<'INNER'
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
INNER
chmod +x "$MANI/run_all_on_worker.sh"

dispatch_one() {
  local i="$1"
  local name="$2"
  local zone="$3"
  local wc="$MANI/worker_${i}.json"
  echo "[dispatch] -> $name (zone=$zone)" | tee -a "$LOG"

  # 1. Attach secrets + run-id as instance metadata (NOT baked into scripts).
  gcloud compute instances add-metadata "$name" --zone="$zone" \
    --metadata="hf-token-secret=${HF_TOKEN_VAL},worker-idx=${i},bucket=${BUCKET},run-id=${RUN_ID}" \
    2>&1 | tail -2 | tee -a "$LOG"

  # 2. Push payload.
  gcloud compute scp --zone="$zone" \
    "$SUB/scripts/worker/bootstrap.sh" \
    "$SUB/scripts/worker/pull_assigned_shards.py" \
    "$SUB/scripts/worker/hash_pass.py" \
    "$SUB/scripts/worker/upload_output.sh" \
    "$SUB/scripts/worker/_done_sentinel.sh" \
    "$REPO/glossapi_corpus_cli/text_dedup.py" \
    "$wc" \
    "$MANI/run_all_on_worker.sh" \
    "${name}:/tmp/payload/" --recurse 2>&1 | tail -3 | tee -a "$LOG"

  # 3. Move payload into place and rename worker_<i>.json → worker_config.json.
  gcloud compute ssh "$name" --zone="$zone" --command="
    sudo mkdir -p /home/foivos/worker &&
    sudo mv /tmp/payload/*.sh /tmp/payload/*.py /tmp/payload/*.json /home/foivos/worker/ &&
    sudo mv /home/foivos/worker/worker_${i}.json /home/foivos/worker/worker_config.json &&
    sudo chown -R foivos:foivos /home/foivos/worker
  " 2>&1 | tail -2 | tee -a "$LOG"

  # 4. Detached run. setsid --fork makes the launcher return after spawning
  #    the worker; the worker writes its own pid from inside the fork.
  local launch_ok=1
  if timeout 90s gcloud compute ssh "$name" --zone="$zone" \
    --ssh-flag="-o ServerAliveInterval=10" \
    --ssh-flag="-o ServerAliveCountMax=3" \
    --command="
    chmod +x /home/foivos/worker/run_all_on_worker.sh &&
    setsid --fork bash -c 'echo \$\$ > /home/foivos/worker/run_all.pid; exec /home/foivos/worker/run_all_on_worker.sh' </dev/null >/home/foivos/worker/run_all.out 2>&1 &&
    echo 'launched $name'
  " 2>&1 | tail -2 | tee -a "$LOG"; then
    launch_ok=0
  else
    echo "[dispatch] FATAL: launch command timed out/failed for $name" | tee -a "$LOG" >&2
  fi

  # Give the detached worker a short window to read metadata, then remove the
  # secret so it is not left visible for the lifetime of the VM.
  sleep 20
  gcloud compute instances remove-metadata "$name" --zone="$zone" \
    --keys=hf-token-secret 2>&1 | tail -2 | tee -a "$LOG" || true
  if [ "$launch_ok" -ne 0 ]; then
    return 7
  fi
}

# Iterate over workers.list — TSV: <name>\t<zone>. Line index = logical worker_idx.
WORKERS_LIST="$MANI/workers.list"
if [ ! -f "$WORKERS_LIST" ]; then
  echo "[dispatch] FATAL: $WORKERS_LIST missing — run 02_spin_up_workers.sh first" >&2
  exit 1
fi
LINE_IDX=0
pids=()
while IFS=$'\t' read -r WORKER_NAME WORKER_ZONE; do
  [ -z "$WORKER_NAME" ] && continue
  dispatch_one "$LINE_IDX" "$WORKER_NAME" "$WORKER_ZONE" &
  pids+=("$!")
  LINE_IDX=$((LINE_IDX + 1))
done < "$WORKERS_LIST"
FAILED=0
for pid in "${pids[@]}"; do
  wait "$pid" || FAILED=1
done
if [ "$FAILED" -ne 0 ]; then
  echo "[dispatch] FATAL: one or more worker launches failed" | tee -a "$LOG" >&2
  exit 7
fi

echo "=== dispatch done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
echo "[dispatch] supervisor: run scripts/coordinator/04_poll_and_collect.py"
