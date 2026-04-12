#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <working_release_root> <hplt_release_root>" >&2
  exit 2
fi

WORKING_RELEASE_ROOT="$1"
HPLT_RELEASE_ROOT="$2"
SUMMARY_JSON="${HPLT_RELEASE_ROOT}/hplt_build_summary_ge8_no_mt.json"
INTEGRATE_SCRIPT="${HOME}/Projects/glossapi-tokenizer-extension/subprojects/01_hplt_filtering/scripts/integrate_hplt_slice_into_working_release.py"
PYTHON_BIN="${HOME}/venvs/glossapi-corpus-clean/bin/python"

while [ ! -f "${SUMMARY_JSON}" ]; do
  sleep 60
done

exec "${PYTHON_BIN}" "${INTEGRATE_SCRIPT}" \
  --working-release-root "${WORKING_RELEASE_ROOT}" \
  --hplt-release-root "${HPLT_RELEASE_ROOT}"
