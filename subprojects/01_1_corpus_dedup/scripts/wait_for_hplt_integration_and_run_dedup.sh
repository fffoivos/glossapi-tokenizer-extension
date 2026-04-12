#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <working_release_root> <state_root>" >&2
  exit 2
fi

WORKING_RELEASE_ROOT="$1"
STATE_ROOT="$2"
INTEGRATION_SUMMARY="${WORKING_RELEASE_ROOT}/hplt_integration_summary.json"
PYTHON_BIN="${HOME}/venvs/glossapi-corpus-clean/bin/python"
export PYTHONPATH="${HOME}/data/glossapi_work"
export GLOSSAPI_WORK_ROOT="${HOME}/data/glossapi_work"

while [ ! -f "${INTEGRATION_SUMMARY}" ]; do
  sleep 60
done

exec "${PYTHON_BIN}" -m glossapi_corpus_cli.cli dedup-text run \
  --input-root "${WORKING_RELEASE_ROOT}/data" \
  --state-root "${STATE_ROOT}" \
  --max-workers 32
