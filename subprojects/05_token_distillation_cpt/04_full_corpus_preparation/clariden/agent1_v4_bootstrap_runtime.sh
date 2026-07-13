#!/usr/bin/env bash
# Build a v4-specific immutable runtime on an allocated Clariden CPU node.
set -euo pipefail

[[ ! -e "$AGENT1_V4_RUNTIME_VENV" ]] || { echo "ERROR: immutable v4 runtime exists" >&2; exit 20; }
[[ -s "$AGENT1_V4_ENVIRONMENT_LOCK" ]] || { echo "ERROR: runtime requirements are missing" >&2; exit 20; }
partial="${AGENT1_V4_RUNTIME_VENV}.partial-${SLURM_JOB_ID}"
[[ ! -e "$partial" ]] || { echo "ERROR: stale runtime partial exists" >&2; exit 20; }
mkdir -p "$(dirname "$AGENT1_V4_RUNTIME_VENV")"

uenv run "$AGENT1_V4_UENV" --view=default -- bash -lc '
set -euo pipefail
test "$(uname -m)" = aarch64
python3 -m venv "$1"
"$1/bin/python" -m pip install --requirement "$2"
"$1/bin/python" - <<"PY"
import duckdb, pyarrow
print("agent1-v4-runtime", pyarrow.__version__, duckdb.__version__)
PY
' bash "$partial" "$AGENT1_V4_ENVIRONMENT_LOCK"
mv "$partial" "$AGENT1_V4_RUNTIME_VENV"
uenv run "$AGENT1_V4_UENV" --view=default -- \
    "$AGENT1_V4_RUNTIME_VENV/bin/python" - "$AGENT1_V4_ATTEMPT_DIR/runtime_receipt.json" <<'PY'
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "agent1_v4_runtime_receipt_v1",
    "machine": platform.machine(),
    "packages": {name: version(name) for name in ("pyarrow", "duckdb")},
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
ln -s "../receipts/00_bootstrap/attempts/${SLURM_JOB_ID}/runtime_receipt.json" "$AGENT1_V4_RUN_ROOT/00_frozen/runtime_receipt.json"
