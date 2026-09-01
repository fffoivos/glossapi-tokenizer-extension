#!/usr/bin/env bash
# Read-only Mac-side status recorder for the active full-8B campaign.
# It submits no Slurm work and never changes a remote file or job state.
# Its retention summary is advisory: the immutable training/supervisor gates
# remain the authority for checkpointing and campaign decisions.
set -euo pipefail
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH

interval_seconds=${FULL8_CAMPAIGN_WATCH_INTERVAL_SECONDS:-120}
max_checks=${FULL8_CAMPAIGN_WATCH_MAX_CHECKS:-3000}
[[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid interval" >&2
  exit 2
}
[[ "$max_checks" =~ ^[1-9][0-9]*$ ]] || {
  echo "invalid max checks" >&2
  exit 2
}

for ((check=1; check<=max_checks; check++)); do
  if result=$(ssh -o BatchMode=yes -o ConnectTimeout=15 clariden bash -s <<'REMOTE'
set -euo pipefail
run=/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/20260808T121000Z-d0-wsd10-sanitized-successor-v12
if [[ -f "$run/campaign_evidence_completion_receipt.json" ]]; then
  printf 'DONE terminal_receipt=%s training_receipt=%s\n' \
    "$run/campaign_evidence_completion_receipt.json" \
    "$run/training_completion_receipt.json"
  exit 0
fi
if [[ -f "$run/training_completion_receipt.json" ]]; then
  printf 'FINALIZING training_receipt=%s campaign_evidence_receipt=pending\n' \
    "$run/training_completion_receipt.json"
fi
jobs=$(squeue -h -u fffoivos -o '%i|%j|%T|%P|%D|%M|%L|%R' \
  | awk -F'|' '$2 ~ /^full8b/ {print}' | sort | tr '\n' ';')
latest_log=$(find "$run/logs" -maxdepth 1 -type f -name 'full8b_s*-*.out' \
  -printf '%T@ %f\n' 2>/dev/null | sort -n | tail -n 1 || true)
active_train=''
train_log=''
train_err=''
latest_metric=''
latest_metric_epoch=-1
# A permit-gated successor holder is also a RUNNING normal full8b_s* job while
# it waits.  Treat a job as the active training leaf only after its canonical
# log contains a real Megatron iteration metric.  This preserves read-only
# monitoring across the short interval where a live segment and its successor
# holder coexist, without relying on an ambiguous Slurm job name.
candidate_rows=$(squeue -h -u fffoivos -o '%i|%j|%T|%P' \
  | awk -F'|' '$3 == "RUNNING" && $4 == "normal" && $2 ~ /^full8b_s[0-9]+a[0-9]+(_hold)?$/ {print}')
while IFS='|' read -r candidate_job candidate_name _candidate_state _candidate_partition; do
  [[ -n "${candidate_job:-}" ]] || continue
  candidate_log="$run/logs/${candidate_name}-${candidate_job}.out"
  candidate_metric=$(grep -E 'iteration[[:space:]]+[0-9]+/' "$candidate_log" 2>/dev/null \
    | tail -n 1 | tr '\n' ' ' || true)
  [[ -n "$candidate_metric" ]] || continue
  candidate_metric_epoch=$(stat -c %Y "$candidate_log" 2>/dev/null || printf '0')
  [[ "$candidate_metric_epoch" =~ ^[0-9]+$ ]] || candidate_metric_epoch=0
  (( candidate_metric_epoch > latest_metric_epoch )) || continue
  active_train="${candidate_job}|${candidate_name}|${_candidate_state}|${_candidate_partition}"
  train_log="$candidate_log"
  train_err="$run/logs/${candidate_name}-${candidate_job}.err"
  latest_metric="$candidate_metric"
  latest_metric_epoch="$candidate_metric_epoch"
done <<<"$candidate_rows"
latest_stderr=$(tail -n 1 "$train_err" 2>/dev/null | tr '\n' ' ' || true)
stage_recipe=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/full8_mixed_sanitized/20260808T064500Z-d0-v4-v45bridge/contracts/rebind_v3/recipe_8b_full_mixed.sanitized.json
if retention=$(/usr/bin/python3.11 - "$run/logs" "$stage_recipe" <<'PY'
import json
import math
import re
import sys
from pathlib import Path

log_root = Path(sys.argv[1])
recipe_path = Path(sys.argv[2])
recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
policy = recipe["evaluation"]["retention_alerts"]
panels = tuple(policy["panels"])
reference = int(policy["reference"]["start_update"])
warning_limit = float(policy["warning"]["any_panel_increase_nats"])
critical_limit = float(policy["critical"]["any_panel_increase_nats"])
macro_limit = float(policy["critical"]["macro_mean_increase_nats"])
required = int(policy["critical"]["consecutive_observations"])
pattern = re.compile(
    r"validation loss at iteration\s+(\d+)\s+\[([^\]]+)\]"
    r"\s+\|\s+lm loss value:\s+([0-9.+-Ee]+)"
)
selected = {}
for path in sorted(log_root.glob("full8b_s*-*.out")):
    for iteration, panel, loss in pattern.findall(path.read_text(errors="replace")):
        iteration = int(iteration)
        if panel not in panels or iteration < reference:
            continue
        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError("nonfinite_validation_loss")
        key = (iteration, panel)
        previous = selected.get(key)
        if previous is not None and abs(previous - loss) > 1.0e-9:
            raise ValueError("conflicting_duplicate_validation")
        selected[key] = loss
points = []
for iteration in sorted({iteration for iteration, _panel in selected}):
    row = {panel: selected.get((iteration, panel)) for panel in panels}
    if all(value is not None for value in row.values()):
        points.append((iteration, row))
if not points:
    print("RETENTION " + json.dumps({"status": "pending"}, sort_keys=True))
    raise SystemExit(0)
mins = {panel: math.inf for panel in panels}
warning_runs = {panel: 0 for panel in panels}
critical_runs = {panel: 0 for panel in panels}
warning_max = {panel: 0 for panel in panels}
critical_max = {panel: 0 for panel in panels}
macro_run = macro_max = 0
last_deltas = {}
last_macro = 0.0
for iteration, row in points:
    deltas = {}
    for panel in panels:
        mins[panel] = min(mins[panel], row[panel])
        delta = row[panel] - mins[panel]
        deltas[panel] = delta
        warning_runs[panel] = warning_runs[panel] + 1 if delta >= warning_limit else 0
        critical_runs[panel] = critical_runs[panel] + 1 if delta >= critical_limit else 0
        warning_max[panel] = max(warning_max[panel], warning_runs[panel])
        critical_max[panel] = max(critical_max[panel], critical_runs[panel])
    macro = sum(deltas.values()) / len(panels)
    macro_run = macro_run + 1 if macro >= macro_limit else 0
    macro_max = max(macro_max, macro_run)
    last_deltas = deltas
    last_macro = macro
warning_panels = [panel for panel in panels if warning_max[panel] >= required]
critical_panels = [panel for panel in panels if critical_max[panel] >= required]
status = "critical" if critical_panels or macro_max >= required else "warning" if warning_panels else "no_alert"
payload = {
    "status": status,
    "iteration": points[-1][0],
    "complete_points": len(points),
    "macro_delta_nats": round(last_macro, 6),
    "current_warning_candidates": {
        panel: round(delta, 6) for panel, delta in last_deltas.items()
        if delta >= warning_limit
    },
    "warning_panels": warning_panels,
    "critical_panels": critical_panels,
}
print("RETENTION " + json.dumps(payload, sort_keys=True, separators=(",", ":")))
PY
); then
  :
else
  retention='RETENTION {"status":"unavailable"}'
fi
printf 'RUN jobs=%s active_train=%s latest_log=%s latest_metric=%s latest_stderr=%s retention=%s\n' \
  "${jobs:-none}" "${active_train:-none}" "${latest_log:-none}" \
  "${latest_metric:-none}" "${latest_stderr:-none}" "$retention"
REMOTE
  ); then
    :
  else
    result="WAIT ssh_or_remote_error"
  fi
  printf '%s check=%s %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$check" "$result"
  [[ "$result" == DONE* ]] && exit 0
  sleep "$interval_seconds"
done

echo "watch exhausted after $max_checks checks" >&2
exit 1
