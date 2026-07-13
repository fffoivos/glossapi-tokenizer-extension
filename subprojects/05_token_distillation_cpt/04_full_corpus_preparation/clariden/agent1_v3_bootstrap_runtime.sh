#!/usr/bin/env bash
# Build an immutable, AArch64-only Python runtime for one Agent 1 v3 run.
set -euo pipefail

: "${PHASE04_DIR:?PHASE04_DIR must be exported by the v3 dispatcher}"
: "${AGENT1_V3_UENV:?AGENT1_V3_UENV must be exported by the v3 dispatcher}"
[[ ! -e "$AGENT1_V3_RUNTIME_VENV" ]] || {
    echo "ERROR: immutable v3 runtime already exists: $AGENT1_V3_RUNTIME_VENV" >&2
    exit 20
}
requirements="$PHASE04_DIR/requirements-runtime.txt"
test -s "$requirements"
partial="${AGENT1_V3_RUNTIME_VENV}.partial-${SLURM_JOB_ID}"
[[ ! -e "$partial" ]] || { echo "ERROR: stale runtime partial exists: $partial" >&2; exit 20; }
mkdir -p "$(dirname "$AGENT1_V3_RUNTIME_VENV")"

uenv run "$AGENT1_V3_UENV" --view=default -- bash -lc '
set -euo pipefail
test "$(uname -m)" = aarch64
python3 -m venv "$1"
"$1/bin/python" -m pip install --requirement "$2"
"$1/bin/python" - <<"PY"
import duckdb, huggingface_hub, pyarrow, tokenizers
print("agent1-v3-runtime", pyarrow.__version__, duckdb.__version__, tokenizers.__version__, huggingface_hub.__version__)
PY
' bash "$partial" "$requirements"

mv "$partial" "$AGENT1_V3_RUNTIME_VENV"
uenv run "$AGENT1_V3_UENV" --view=default -- \
    "$AGENT1_V3_RUNTIME_VENV/bin/python" - "$AGENT1_V3_ATTEMPT_DIR/runtime_receipt.json" <<'PY'
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": "agent1_full_corpus_v3_runtime_receipt_v1",
    "machine": platform.machine(),
    "packages": {name: version(name) for name in ("pyarrow", "duckdb", "tokenizers", "huggingface-hub")},
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
ln -s "attempts/$AGENT1_V3_ATTEMPT_ID/runtime_receipt.json" "$AGENT1_V3_PHASE0_DIR/runtime_receipt.json"
echo "AGENT1_V3_RUNTIME=$AGENT1_V3_RUNTIME_VENV"
