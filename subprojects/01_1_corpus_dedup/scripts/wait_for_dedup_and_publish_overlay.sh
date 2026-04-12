#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <working_release_root> <state_root>" >&2
  exit 2
fi

WORKING_RELEASE_ROOT="$1"
STATE_ROOT="$2"
LATEST_SUCCESS_JSON="${STATE_ROOT}/latest_success.json"
PYTHON_BIN="${HOME}/venvs/glossapi-corpus-clean/bin/python"
PUBLISH_SCRIPT="${HOME}/Projects/glossapi-tokenizer-extension/subprojects/01_1_corpus_dedup/scripts/publish_dedup_overlay_into_working_release.py"

while [ ! -f "${LATEST_SUCCESS_JSON}" ]; do
  sleep 60
done

exec "${PYTHON_BIN}" "${PUBLISH_SCRIPT}" \
  --working-release-root "${WORKING_RELEASE_ROOT}" \
  --state-root "${STATE_ROOT}"
