#!/usr/bin/env bash
# Create the Phase-04 aarch64 runtime. This prepares software only; it submits no jobs.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "runtime bootstrap"
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"

command -v uenv >/dev/null || { echo "ERROR: uenv not found" >&2; exit 2; }
test -s "$PHASE04_DIR/requirements-runtime.txt"

if [[ -e "$RUNTIME_VENV" && "${REBUILD_RUNTIME:-0}" != "1" ]]; then
    echo "ERROR: runtime already exists: $RUNTIME_VENV" >&2
    echo "Set REBUILD_RUNTIME=1 to replace it atomically." >&2
    exit 3
fi

PARTIAL="${RUNTIME_VENV}.partial-$$"
BACKUP="${RUNTIME_VENV}.previous-$$"
[[ ! -e "$PARTIAL" && ! -e "$BACKUP" ]] || {
    echo "ERROR: stale runtime staging path exists: $PARTIAL or $BACKUP" >&2
    exit 4
}
mkdir -p "$(dirname "$RUNTIME_VENV")"
cleanup() { rm -rf "$PARTIAL"; }
trap cleanup EXIT

echo "Preparing an aarch64 runtime at $PARTIAL inside $PHASE04_UENV"
uenv run "$PHASE04_UENV" --view=default -- bash -lc '
set -euo pipefail
[[ "$(uname -m)" == "aarch64" ]] || { echo "ERROR: uenv runtime is not aarch64" >&2; exit 5; }
python3 -m venv "$1"
"$1/bin/python" -m pip install --requirement "$2"
"$1/bin/python" - <<"PY"
import blake3, datasets, duckdb, huggingface_hub, pyarrow, regex, tokenizers, zstandard
print(
    "python runtime OK",
    pyarrow.__version__,
    tokenizers.__version__,
    huggingface_hub.__version__,
    duckdb.__version__,
    zstandard.__version__,
)
PY
"$1/bin/python" "$3" --help >/dev/null
' bash "$PARTIAL" "$PHASE04_DIR/requirements-runtime.txt" "$PHASE04_DIR/scripts/invoke_text_dedup.py"

if [[ -e "$RUNTIME_VENV" ]]; then
    mv "$RUNTIME_VENV" "$BACKUP"
fi
if ! mv "$PARTIAL" "$RUNTIME_VENV"; then
    [[ ! -e "$BACKUP" ]] || mv "$BACKUP" "$RUNTIME_VENV"
    exit 6
fi
rm -rf "$BACKUP"
trap - EXIT

echo "Runtime ready: $RUNTIME_VENV"
