#!/usr/bin/env bash
# Coordinator step 99: tear down all workers + joins-worker for this run.
# Compute SA cannot self-delete (gcloud_compute_sa_no_delete.md); the
# coordinator MUST do this. Verifies zero remaining instances at end.
#
# Multi-zone aware (per 2026-05-18 lessons): looks up each instance's zone from
# `gcloud instances list` rather than hardcoding europe-west4-b, so workers
# spun up via zone-fallback (-c, -b, -a) are all cleaned up.
#
# Auth-aware: prints a clear "AUTH FAILED" message if the listing call errors
# (rather than misinterpreting an empty list as "nothing to delete"). The
# 2026-05-18 run had a stale OAuth token that caused 99 to report "no
# instances found" while 4 workers were still RUNNING and burning $.
set -euo pipefail

SUB="/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit"
RUN_ID="${RUN_ID:-$(cat "$SUB/manifests/CURRENT_RUN_ID")}"
RUN_LOWER="${RUN_ID,,}"
MANI="$SUB/manifests/run_$RUN_ID"
LOG="$MANI/teardown_log.txt"

echo "=== teardown start $(date -u +%Y-%m-%dT%H:%M:%SZ) RUN_ID=$RUN_ID ===" | tee -a "$LOG"

# Pre-flight: confirm we have a working auth token. Don't silently treat an
# auth failure as "nothing to delete".
if ! gcloud auth print-access-token >/dev/null 2>&1; then
  echo "[teardown] FATAL: gcloud auth token unavailable. Run 'gcloud auth login' first." | tee -a "$LOG" >&2
  exit 3
fi

# List <name>\t<zone> rows matching the run label, across all zones.
NAMES_ZONES=$(gcloud compute instances list \
                --filter="labels.workload=apertus-c3-dedup AND labels.run=${RUN_LOWER}" \
                --format="value(name,zone)" 2>"$MANI/.teardown_list_err.tmp" || true)
LIST_ERR=$(cat "$MANI/.teardown_list_err.tmp" 2>/dev/null || true)
rm -f "$MANI/.teardown_list_err.tmp"
if [ -n "$LIST_ERR" ] && echo "$LIST_ERR" | grep -qiE "(error|denied|reauth)"; then
  echo "[teardown] FATAL: instances list returned an error: $LIST_ERR" | tee -a "$LOG" >&2
  exit 4
fi

if [ -z "$NAMES_ZONES" ]; then
  echo "[teardown] no instances found with workload=apertus-c3-dedup AND run=${RUN_LOWER}" | tee -a "$LOG"
else
  echo "[teardown] deleting:" | tee -a "$LOG"
  echo "$NAMES_ZONES" | tee -a "$LOG"
  # Parse name\tzone (gcloud value() emits a long zone URL — extract last segment).
  while IFS=$'\t' read -r NAME ZONE_URL; do
    [ -z "$NAME" ] && continue
    ZONE_SHORT="${ZONE_URL##*/}"
    gcloud compute instances delete "$NAME" --zone="$ZONE_SHORT" --quiet 2>&1 | tee -a "$LOG" &
  done <<<"$NAMES_ZONES"
  wait
fi

# Verify zero remaining.
REMAINING=$(gcloud compute instances list \
              --filter="labels.workload=apertus-c3-dedup AND labels.run=${RUN_LOWER}" \
              --format="value(name)" 2>/dev/null | grep -c . || true)
echo "[teardown] remaining instances with this run label: $REMAINING" | tee -a "$LOG"
if [ "$REMAINING" -ne 0 ]; then
  echo "[teardown] ERROR — verification FAILED; manual cleanup needed" | tee -a "$LOG"
  exit 2
fi
echo "=== teardown done $(date -u +%Y-%m-%dT%H:%M:%SZ) — zero remaining ===" | tee -a "$LOG"
