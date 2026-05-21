#!/usr/bin/env bash
# Coordinator step 02: create 8 × c4-highcpu-192 SPOT workers with zone fallback.
# Try ZONES[0] first; any STOCKOUT-failed slots retry in ZONES[1..N].
# This is the COST-EVENT step. Only invoke after pre-flight + partition pass.
#
# Inputs: $RUN_ID env (or read from CURRENT_RUN_ID)
# Outputs: up to 8 RUNNING instances, labelled
#          workload=apertus-c3-dedup, run=<run_id>, worker-idx=<0..7>, owner=foivos
#          + manifests/run_<RUN_ID>/workers.list — TSV: <name>\t<zone>
#          + per-worker external IP added (SSH from home; project has no IAP firewall).
#
# Per 2026-05-18 lessons:
# - europe-west4-b ran out of c4-highcpu-192 SPOT capacity mid-run (STOCKOUT).
# - europe-west4-c had capacity per the GCP error message. We try -c first.
# - `--no-address` would require IAP; the project doesn't have it set up.
#   So we just create with default external IP (gcloud's default).
# - workers.list now carries zone so dispatch + teardown work across zones.
set -euo pipefail

SUB="/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit"
RUN_ID="${RUN_ID:-$(cat "$SUB/manifests/CURRENT_RUN_ID")}"
RUN_LOWER="${RUN_ID,,}"
NAME_SUFFIX="${RUN_LOWER//_/-}"
MANI="$SUB/manifests/run_$RUN_ID"
MACHINE="c4-highcpu-192"
IMAGE_FAMILY="debian-12"
IMAGE_PROJECT="debian-cloud"
ZONES=("europe-west4-c" "europe-west4-b" "europe-west4-a")
WORKER_COUNT="${WORKER_COUNT:-8}"

LOG="$MANI/spin_up.log"
echo "=== spin up start $(date -u +%Y-%m-%dT%H:%M:%SZ) RUN_ID=$RUN_ID workers=$WORKER_COUNT ===" | tee -a "$LOG"
echo "=== zones tried in order: ${ZONES[*]} ===" | tee -a "$LOG"

# Track results per worker idx: created_zone[i]= "<zone>" if created, else empty.
declare -A CREATED_ZONE

try_create_one() {
  local i="$1"
  local zone="$2"
  local name="dedup-w${i}-${NAME_SUFFIX}"
  local rc_file="$MANI/.create_rc_${i}_${zone}.tmp"
  echo "[spin_up] zone=${zone} create $name (worker_idx=$i) ..." | tee -a "$LOG"
  if gcloud compute instances create "$name" \
       --zone="$zone" \
       --machine-type="$MACHINE" \
       --provisioning-model=SPOT \
       --instance-termination-action=DELETE \
       --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
       --boot-disk-size=500GB --boot-disk-type=hyperdisk-balanced \
       --scopes=cloud-platform \
       --labels=workload=apertus-c3-dedup,run="${RUN_LOWER}",worker-idx="$i",owner=foivos \
       --metadata="worker-idx=$i,run-id=$RUN_ID" \
       2>>"$LOG" >>"$LOG"; then
    echo "$zone" > "$rc_file"
  else
    echo "" > "$rc_file"
  fi
}

# Initialise pending list.
PENDING=()
for i in $(seq 0 $((WORKER_COUNT - 1))); do PENDING+=("$i"); done

for zone in "${ZONES[@]}"; do
  [ ${#PENDING[@]} -eq 0 ] && break
  echo "[spin_up] trying zone=$zone for ${#PENDING[@]} pending worker(s)" | tee -a "$LOG"
  ROUND_PENDING=("${PENDING[@]}")
  PENDING=()
  for i in "${ROUND_PENDING[@]}"; do
    try_create_one "$i" "$zone" &
  done
  wait
  for i in "${ROUND_PENDING[@]}"; do
    rc_file="$MANI/.create_rc_${i}_${zone}.tmp"
    if [ -s "$rc_file" ]; then
      CREATED_ZONE[$i]="$zone"
    else
      PENDING+=("$i")
    fi
    rm -f "$rc_file"
  done
  echo "[spin_up] after zone=$zone: created=${#CREATED_ZONE[@]}  still_pending=${#PENDING[@]}" | tee -a "$LOG"
done

# Write workers.list — TSV: <name>\t<zone>. Sorted by worker idx for deterministic dispatch.
: > "$MANI/workers.list"
for i in $(seq 0 $((WORKER_COUNT - 1))); do
  z="${CREATED_ZONE[$i]:-}"
  if [ -n "$z" ]; then
    echo -e "dedup-w${i}-${NAME_SUFFIX}\t${z}" >> "$MANI/workers.list"
  fi
done
N_CREATED=$(wc -l <"$MANI/workers.list")
echo "[spin_up] workers.list (${N_CREATED} workers, name\\tzone):" | tee -a "$LOG"
cat "$MANI/workers.list" | tee -a "$LOG"

# HARD GATE: must have created exactly WORKER_COUNT workers across all fallback zones.
# A partial creation would force us to either repartition (extra work) or run with
# silent under-coverage. The 2026-05-18 run was a partial that we ad-hoc fixed by
# repartitioning to 4 workers — that's an explicit decision the operator should make,
# not a silent fallback.
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
if [ "$N_CREATED" -ne "$WORKER_COUNT" ]; then
  if [ "$ALLOW_PARTIAL" = "1" ]; then
    echo "[spin_up] WARN: only ${N_CREATED}/${WORKER_COUNT} created; ALLOW_PARTIAL=1, continuing. Operator MUST repartition with WORKER_COUNT=${N_CREATED} before dispatch." | tee -a "$LOG"
  else
    echo "[spin_up] FATAL: only ${N_CREATED}/${WORKER_COUNT} workers created across zones ${ZONES[*]}." | tee -a "$LOG" >&2
    echo "[spin_up] Options: (a) wait + retry, (b) re-run with ALLOW_PARTIAL=1 + repartition with WORKER_COUNT=${N_CREATED} ./01_bytes_balanced_partition.py" | tee -a "$LOG" >&2
    exit 5
  fi
fi

# Wait for all created workers to reach RUNNING. Hard-fail if deadline expires.
echo "[spin_up] waiting for all ${N_CREATED} instances to reach RUNNING ..." | tee -a "$LOG"
DEADLINE=$(( $(date +%s) + 600 ))
TARGET="${N_CREATED}"
ALL_RUNNING=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  STATUSES=$(gcloud compute instances list \
              --filter="labels.run=${RUN_LOWER}" \
              --format="value(status)" 2>/dev/null | sort | uniq -c | tr -s ' ' || true)
  echo "  $(date -u +%H:%M:%S)  $STATUSES" | tee -a "$LOG"
  if echo "$STATUSES" | grep -q "${TARGET} RUNNING"; then
    echo "[spin_up] all ${TARGET} RUNNING" | tee -a "$LOG"
    ALL_RUNNING=1
    break
  fi
  sleep 30
done
if [ "$ALL_RUNNING" -ne 1 ]; then
  echo "[spin_up] FATAL: RUNNING deadline (10 min) expired with fewer than ${TARGET} RUNNING. Some workers stuck in PROVISIONING/STAGING — likely spot-preempted before boot. Tear down + retry." | tee -a "$LOG" >&2
  exit 6
fi
echo "=== spin up done $(date -u +%Y-%m-%dT%H:%M:%SZ) ($N_CREATED / $WORKER_COUNT created, all RUNNING) ===" | tee -a "$LOG"
