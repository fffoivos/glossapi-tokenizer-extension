#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <working_release_root> <state_root>" >&2
  exit 2
fi

WORKING_RELEASE_ROOT="$1"
STATE_ROOT="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
LATEST_SUCCESS_JSON="${STATE_ROOT}/latest_success.json"
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
  echo "Could not resolve a usable Python interpreter for dedup overlay publish" >&2
  exit 1
fi
PUBLISH_SCRIPT="${REPO_ROOT}/subprojects/01_1_corpus_dedup/scripts/publish_dedup_overlay_into_working_release.py"

while [ ! -f "${LATEST_SUCCESS_JSON}" ]; do
  sleep 60
done

exec "${PYTHON_BIN}" "${PUBLISH_SCRIPT}" \
  --working-release-root "${WORKING_RELEASE_ROOT}" \
  --state-root "${STATE_ROOT}"
