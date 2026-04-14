#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <working_release_root> <handoff_root>" >&2
  exit 2
fi

WORKING_RELEASE_ROOT="$1"
HANDOFF_ROOT="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HPLT_INTEGRATION_SUMMARY="${WORKING_RELEASE_ROOT}/hplt_integration_summary.json"
DATA_ROOT="${WORKING_RELEASE_ROOT}/data"
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
  echo "Could not resolve a usable Python interpreter for source-only uploader handoff prep" >&2
  exit 1
fi
PREPARE_SCRIPT="${REPO_ROOT}/ops/upload/prepare_hf_uploader_handoff.py"
REPO_ID="${HF_RELEASE_REPO_ID:-fffoivos/glossapi-greek-nanochat-pretraining-dataset}"
VISIBILITY_FLAG="${HF_RELEASE_VISIBILITY_FLAG:---public}"
REMOTE_HOST="${HF_UPLOAD_REMOTE_HOST:-}"
REMOTE_USER="${HF_UPLOAD_REMOTE_USER:-}"
REMOTE_RELEASE_ROOT="${HF_UPLOAD_REMOTE_RELEASE_ROOT:-}"
REMOTE_REPO_ROOT="${HF_UPLOAD_REMOTE_REPO_ROOT:-}"
REMOTE_PYTHON="${HF_UPLOAD_REMOTE_PYTHON:-}"
REMOTE_DETACH_BIN="${HF_UPLOAD_REMOTE_DETACH_BIN:-}"
REMOTE_UNIT_PREFIX="${HF_UPLOAD_REMOTE_UNIT_PREFIX:-}"
PUBLISH_SCRIPT_PATH="${HF_UPLOAD_PUBLISH_SCRIPT_PATH:-}"
NUM_WORKERS="${HF_UPLOAD_NUM_WORKERS:-}"
PRINT_REPORT_EVERY="${HF_UPLOAD_PRINT_REPORT_EVERY:-}"
USE_HF_XET_HIGH_PERFORMANCE="${HF_UPLOAD_USE_XET_HIGH_PERFORMANCE:-0}"

while [ ! -d "${DATA_ROOT}" ] || [ ! -f "${HPLT_INTEGRATION_SUMMARY}" ]; do
  sleep 60
done

CMD=(
  "${PYTHON_BIN}"
  "${PREPARE_SCRIPT}"
  --working-release-root "${WORKING_RELEASE_ROOT}"
  --handoff-root "${HANDOFF_ROOT}"
  --repo-id "${REPO_ID}"
  "${VISIBILITY_FLAG}"
  --source-only
)

if [ -n "${REMOTE_HOST}" ]; then
  CMD+=(--remote-host "${REMOTE_HOST}")
fi
if [ -n "${REMOTE_USER}" ]; then
  CMD+=(--remote-user "${REMOTE_USER}")
fi
if [ -n "${REMOTE_RELEASE_ROOT}" ]; then
  CMD+=(--remote-release-root "${REMOTE_RELEASE_ROOT}")
fi
if [ -n "${REMOTE_REPO_ROOT}" ]; then
  CMD+=(--remote-repo-root "${REMOTE_REPO_ROOT}")
fi
if [ -n "${REMOTE_PYTHON}" ]; then
  CMD+=(--remote-python "${REMOTE_PYTHON}")
fi
if [ -n "${REMOTE_DETACH_BIN}" ]; then
  CMD+=(--remote-detach-bin "${REMOTE_DETACH_BIN}")
fi
if [ -n "${REMOTE_UNIT_PREFIX}" ]; then
  CMD+=(--remote-unit-prefix "${REMOTE_UNIT_PREFIX}")
fi
if [ -n "${PUBLISH_SCRIPT_PATH}" ]; then
  CMD+=(--publish-script-path "${PUBLISH_SCRIPT_PATH}")
fi
if [ -n "${NUM_WORKERS}" ]; then
  CMD+=(--num-workers "${NUM_WORKERS}")
fi
if [ -n "${PRINT_REPORT_EVERY}" ]; then
  CMD+=(--print-report-every "${PRINT_REPORT_EVERY}")
fi
if [ "${USE_HF_XET_HIGH_PERFORMANCE}" = "1" ]; then
  CMD+=(--use-hf-xet-high-performance)
fi

exec "${CMD[@]}"
