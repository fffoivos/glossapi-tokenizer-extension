#!/usr/bin/env bash
# Refresh the structured JSON and rendered Markdown status for the 5B report.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBPROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_JSON="${STATE_JSON:-$SUBPROJECT_DIR/reports/latest_5b_report_state.json}"
STATUS_MD="${STATUS_MD:-$SUBPROJECT_DIR/reports/latest_5b_report_status.md}"

cd "$SUBPROJECT_DIR"

python3 -m py_compile \
  scripts/collect_5b_report_state.py \
  scripts/render_5b_report_status.py

scripts/collect_5b_report_state.py --output "$STATE_JSON"
scripts/render_5b_report_status.py --state "$STATE_JSON" --output "$STATUS_MD"

python3 - "$STATE_JSON" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
state = json.loads(state_path.read_text(encoding="utf-8"))
latest = state.get("latest_training", {})
print("refreshed_state={}".format(state_path))
print("collected_at={}".format(state.get("collected_at_utc", "unknown")))
print(
    "latest_iter={} loss={} skipped={} nan={}".format(
        latest.get("current_iter", "unknown"),
        latest.get("lm_loss", "unknown"),
        latest.get("skipped", "unknown"),
        latest.get("nan", "unknown"),
    )
)
print("counts={}".format(json.dumps(state.get("counts", {}), sort_keys=True)))
PY

printf 'rendered_status=%s\n' "$STATUS_MD"
