#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <working_release_root> <hplt_release_root>" >&2
  exit 2
fi

WORKING_RELEASE_ROOT="$1"
HPLT_RELEASE_ROOT="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
INTEGRATE_SCRIPT="${REPO_ROOT}/subprojects/01_hplt_filtering/scripts/integrate_hplt_slice_into_working_release.py"
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
  echo "Could not resolve a usable Python interpreter for HPLT integration" >&2
  exit 1
fi
SUMMARY_CANDIDATES=(
  "${HPLT_RELEASE_ROOT}/hplt_clean60_summary.json"
  "${HPLT_RELEASE_ROOT}/hplt_build_summary_ge8_no_mt.json"
)

while true; do
  for summary_json in "${SUMMARY_CANDIDATES[@]}"; do
    if [ -f "${summary_json}" ]; then
      exec "${PYTHON_BIN}" "${INTEGRATE_SCRIPT}" \
        --working-release-root "${WORKING_RELEASE_ROOT}" \
        --hplt-release-root "${HPLT_RELEASE_ROOT}"
    fi
  done

  if find "${HPLT_RELEASE_ROOT}" -maxdepth 1 -type f -name '*summary*.json' | grep -q .; then
    exec "${PYTHON_BIN}" "${INTEGRATE_SCRIPT}" \
      --working-release-root "${WORKING_RELEASE_ROOT}" \
      --hplt-release-root "${HPLT_RELEASE_ROOT}"
  fi

  sleep 60
done
