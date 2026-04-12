#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <repo_id> <local_dir>" >&2
  exit 2
fi

REPO_ID="$1"
LOCAL_DIR="$2"
VENV_ROOT="${HOME}/venvs/hf-transfer"

mkdir -p "${HOME}/venvs"
if [ ! -x "${VENV_ROOT}/bin/python" ]; then
  python3 -m venv "${VENV_ROOT}"
fi

"${VENV_ROOT}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV_ROOT}/bin/python" -m pip install --upgrade huggingface_hub hf_xet >/dev/null

mkdir -p "${LOCAL_DIR}"
export HF_XET_HIGH_PERFORMANCE=1
export HF_TOKEN="${HF_TOKEN:-$(cat "${HOME}/.cache/huggingface/token" 2>/dev/null || true)}"

exec "${VENV_ROOT}/bin/python" - "$REPO_ID" "$LOCAL_DIR" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id = sys.argv[1]
local_dir = sys.argv[2]
token = os.environ.get("HF_TOKEN") or None

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    allow_patterns=["README.md", "data/*", "dedup_metadata/*"],
    resume_download=True,
    max_workers=16,
    token=token,
)

print("SNAPSHOT_DOWNLOAD_DONE")
PY
