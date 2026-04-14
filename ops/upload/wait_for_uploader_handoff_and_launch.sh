#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <handoff_root>" >&2
  exit 2
fi

HANDOFF_ROOT="$1"
HANDOFF_JSON="${HANDOFF_ROOT}/uploader_handoff.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${TOKENIZER_PIPELINE_PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ]; then
  for candidate in \
    "${HOME}/venvs/glossapi-corpus-clean/bin/python" \
    "${HOME}/data/glossapi_work/.venv/bin/python" \
    "$(command -v python3)"; do
    if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [ -z "${PYTHON_BIN}" ] || [ ! -x "${PYTHON_BIN}" ]; then
  echo "Could not resolve a usable Python interpreter for uploader handoff launch" >&2
  exit 1
fi

LAUNCH_SCRIPT="${REPO_ROOT}/ops/upload/launch_hf_uploader_handoff.py"

while [ ! -f "${HANDOFF_JSON}" ]; do
  sleep 60
done

CMD=(
  "${PYTHON_BIN}"
  "${LAUNCH_SCRIPT}"
  --handoff-json "${HANDOFF_JSON}"
)

if [ -n "${UPLOAD_LOCAL_STAGE_ROOT:-}" ]; then
  CMD+=(--local-stage-root "${UPLOAD_LOCAL_STAGE_ROOT}")
fi
if [ "${UPLOAD_SKIP_SYNC:-0}" = "1" ]; then
  CMD+=(--skip-sync)
fi
if [ "${UPLOAD_SKIP_LAUNCH:-0}" = "1" ]; then
  CMD+=(--skip-launch)
fi
if [ "${UPLOAD_DRY_RUN:-0}" = "1" ]; then
  CMD+=(--dry-run)
fi
if [ -n "${UPLOAD_SSH_KEY:-}" ]; then
  CMD+=(--ssh-key "${UPLOAD_SSH_KEY}")
fi

exec "${CMD[@]}"
