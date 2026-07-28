#!/usr/bin/env bash
set -uo pipefail

: "${CONTRACT:?CONTRACT is required}"
: "${PLAN:?PLAN is required}"
: "${TRAIN_SRC:?TRAIN_SRC is required}"
: "${GLOSSAPI_SRC:?GLOSSAPI_SRC is required}"
: "${MODE:?MODE is required}"
NODE_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
NODE_COUNT="${NODE_COUNT:-3}"
SLOTS="${SLOTS:-36}"
THREADS="${THREADS:-8}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.venvs/hfupload/bin/python}"

mapfile -t UNITS < <("$PYTHON_BIN" - "$PLAN" "$NODE_INDEX" "$NODE_COUNT" "$MODE" <<'PY'
import json
import sys

plan = json.load(open(sys.argv[1]))
units = sorted(plan["units"], key=lambda unit: -unit["estimated_chars"])
mode = sys.argv[4]
if mode == "apply":
    units = [unit for unit in units if unit["apply"]]
elif mode != "dry-run":
    raise SystemExit(f"invalid MODE: {mode}")
for unit in units[int(sys.argv[2])::int(sys.argv[3])]:
    print(unit["unit_id"])
PY
)

export PYTHONPATH="$TRAIN_SRC${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
failure=0

run_unit() {
  local unit_id="$1"
  echo "[$(date -u +%FT%TZ)] start $unit_id"
  "$PYTHON_BIN" -m \
    subprojects.05_token_distillation_cpt.02_corpus_preparation.15_clean_academic.production.run_unit \
    --contract "$CONTRACT" --plan "$PLAN" --unit-id "$unit_id" \
    --mode "$MODE" --glossapi-src "$GLOSSAPI_SRC" --threads "$THREADS"
  local status=$?
  echo "[$(date -u +%FT%TZ)] finish $unit_id exit=$status"
  return "$status"
}
export -f run_unit
export PYTHON_BIN CONTRACT PLAN TRAIN_SRC GLOSSAPI_SRC MODE THREADS PYTHONPATH

echo "node $NODE_INDEX/$NODE_COUNT: ${#UNITS[@]} units; $SLOTS processes x $THREADS threads"
for unit_id in "${UNITS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge "$SLOTS" ]; do
    wait -n || failure=1
  done
  run_unit "$unit_id" &
done
while [ -n "$(jobs -rp)" ]; do
  wait -n || failure=1
done
echo "node $NODE_INDEX complete; failure=$failure"
exit "$failure"
