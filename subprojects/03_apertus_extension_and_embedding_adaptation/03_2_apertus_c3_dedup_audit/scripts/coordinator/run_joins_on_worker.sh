#!/usr/bin/env bash
# Coordinator: spin up ONE c4-highcpu-32 in europe-west4-b, install deps, scp
# 05-09 + manifest, run the joins pipeline in-region, scp final artifacts back,
# tear down. Replaces the home-side step 05 download which is bandwidth-limited
# from europe-west4 (~10 MB/s vs ~115 MB/s same-region).
#
# Per 2026-05-18 lessons:
# - c4-highcpu-8 was undersized for vectorized 07; -32 has CPU headroom for any
#   future scaling without becoming wasteful for the few-minute joins phase.
# - Per-worker subdir + recursive glob in 05 (committed).
# - Vectorized numpy reshape in 07 (committed).
# - Use --no-clobber-style sequential per-worker cp for safety; the same-region
#   cp is fast enough that we don't need to engineer fancier parallelism.
set -euo pipefail

SUB="/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit"
REPO="/home/foivos/Projects/glossapi-tokenizer-extension"
RUN_ID="${RUN_ID:-$(cat "$SUB/manifests/CURRENT_RUN_ID")}"
RUN_LOWER="${RUN_ID,,}"
NAME_SUFFIX="${RUN_LOWER//_/-}"
MANI="$SUB/manifests/run_$RUN_ID"
LOG="$MANI/joins_worker.log"
JW_NAME="dedup-joins-${NAME_SUFFIX}"
JW_ZONE="${JW_ZONE:-europe-west4-b}"
# c4-highmem-32: 32 vCPU + 256 GB RAM. Sized for step 07's full-source
# materialisation (minhash_sig + lsh_band_hashes + per-doc metadata dicts).
# 2026-05-18 partial run had 1.84M europarl rows; clean run will have ~10M+
# hplt_clean60 rows where each row carries ~1.5 KB of sig/band/meta state.
# c4-highcpu-32 (64 GB) was reviewer-flagged as too tight; -highmem-32 gives
# 4x headroom while still being ~$2/hr.
JW_MACHINE="${JW_MACHINE:-c4-highmem-32}"
JW_DISK_SIZE="${JW_DISK_SIZE:-1000GB}"

echo "=== joins-worker start $(date -u +%Y-%m-%dT%H:%M:%SZ) RUN_ID=$RUN_ID ===" | tee -a "$LOG"

cleanup() {
  local rc=$?
  echo "[joins] exit handler firing (rc=$rc) $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
  # Delete the joins-worker on any exit path.
  if gcloud compute instances describe "$JW_NAME" --zone="$JW_ZONE" >/dev/null 2>&1; then
    echo "[joins] deleting $JW_NAME ..." | tee -a "$LOG"
    gcloud compute instances delete "$JW_NAME" --zone="$JW_ZONE" --quiet 2>&1 | tail -3 | tee -a "$LOG" || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

# 1. Spin up joins-worker.
echo "[joins] creating $JW_NAME in $JW_ZONE (${JW_MACHINE}) ..." | tee -a "$LOG"
gcloud compute instances create "$JW_NAME" \
  --zone="$JW_ZONE" \
  --machine-type="$JW_MACHINE" \
  --image-family=debian-12 --image-project=debian-cloud \
  --boot-disk-type=hyperdisk-balanced --boot-disk-size="$JW_DISK_SIZE" \
  --scopes=cloud-platform \
  --labels=workload=apertus-c3-dedup,run="${RUN_LOWER}",owner=foivos,phase=joins \
  2>&1 | tail -3 | tee -a "$LOG"

# 2. Wait briefly for SSH-readiness.
echo "[joins] waiting for SSH ..." | tee -a "$LOG"
DEADLINE=$(( $(date +%s) + 180 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if gcloud compute ssh "$JW_NAME" --zone="$JW_ZONE" --command='echo ok' >/dev/null 2>&1; then
    echo "[joins] SSH ready" | tee -a "$LOG"
    break
  fi
  sleep 10
done

# 3. Bootstrap deps + dir layout on joins-worker.
gcloud compute ssh "$JW_NAME" --zone="$JW_ZONE" --command='
set -e
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip python3-venv python3-dev build-essential 2>&1 | tail -3
python3 -m venv ~/venv
source ~/venv/bin/activate
pip install --quiet --upgrade pip wheel
pip install --quiet pyarrow==17.0.0 polars==1.13.0 numpy==2.1.3 regex blake3 zstandard tqdm
python3 -c "import pyarrow,polars,numpy,blake3; print(\"deps OK\")"
mkdir -p ~/subproject/scripts/coordinator ~/subproject/manifests/run_'"$RUN_ID"' ~/subproject/artifacts
' 2>&1 | tail -5 | tee -a "$LOG"

# 4. SCP scripts + manifest.
gcloud compute scp --zone="$JW_ZONE" \
  "$SUB/manifests/CURRENT_RUN_ID" \
  "${JW_NAME}:/home/foivos/subproject/manifests/" 2>&1 | tail -2 | tee -a "$LOG"
gcloud compute scp --zone="$JW_ZONE" \
  "$MANI/text_dedup_pin.json" \
  "$MANI/workers.list" \
  "${JW_NAME}:/home/foivos/subproject/manifests/run_${RUN_ID}/" 2>&1 | tail -2 | tee -a "$LOG"
gcloud compute scp --zone="$JW_ZONE" \
  "$SUB/scripts/coordinator/05_concat_per_source.py" \
  "$SUB/scripts/coordinator/06_exact_overlap_join.py" \
  "$SUB/scripts/coordinator/07_minhash_overlap_lsh.py" \
  "$SUB/scripts/coordinator/08_holdout_contamination_check.py" \
  "$SUB/scripts/coordinator/09_build_summary_report.py" \
  "${JW_NAME}:/home/foivos/subproject/scripts/coordinator/" 2>&1 | tail -2 | tee -a "$LOG"

# 5. Run pipeline. 05+06 are MANDATORY (fail fast on either). 07 (near-dup)
#    is the OOM-risky step — let it fail gracefully and continue to 09 so the
#    user gets exact-overlap results even if memory pressure kills near-dup.
#    09 already handles missing near/*.parquet by reporting 0 near matches.
gcloud compute ssh "$JW_NAME" --zone="$JW_ZONE" --command='
set -e
source ~/venv/bin/activate
cd ~/subproject
echo "=== 05_concat_per_source ==="
time python3 scripts/coordinator/05_concat_per_source.py
echo "=== 06_exact_overlap_join ==="
time python3 scripts/coordinator/06_exact_overlap_join.py
echo "=== 07_minhash_overlap_lsh (OOM-tolerant — failure does not halt chain) ==="
set +e
time python3 scripts/coordinator/07_minhash_overlap_lsh.py
NEAR_RC=$?
set -e
if [ "$NEAR_RC" -ne 0 ]; then
  echo "[joins] WARN: 07_minhash exited rc=$NEAR_RC; near-dup step skipped/failed. Continuing to report."
  echo "07_minhash_overlap_lsh exited rc=$NEAR_RC ($(date -u +%Y-%m-%dT%H:%M:%SZ))" \
    > ~/subproject/artifacts/'"${RUN_ID}"'/near_overlap_FAILED.txt
fi
echo "=== 08_holdout_contamination_check (non-fatal if no holdout list) ==="
time python3 scripts/coordinator/08_holdout_contamination_check.py || true
echo "=== 09_build_summary_report ==="
time python3 scripts/coordinator/09_build_summary_report.py
echo "=== artifacts tree ==="
find ~/subproject/artifacts/ -type f
' 2>&1 | tee -a "$LOG"

# 6. SCP results back.
echo "[joins] copying artifacts back to home ..." | tee -a "$LOG"
gcloud compute scp --zone="$JW_ZONE" --recurse \
  "${JW_NAME}:/home/foivos/subproject/REPORT_${RUN_ID}.md" \
  "${JW_NAME}:/home/foivos/subproject/artifacts/" \
  "$SUB/" 2>&1 | tail -5 | tee -a "$LOG"

echo "[joins] artifacts copied. Final REPORT:" | tee -a "$LOG"
ls -la "$SUB/REPORT_${RUN_ID}.md" 2>&1 | tee -a "$LOG"

echo "=== joins-worker done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
# trap cleanup runs here, deletes joins-worker.
