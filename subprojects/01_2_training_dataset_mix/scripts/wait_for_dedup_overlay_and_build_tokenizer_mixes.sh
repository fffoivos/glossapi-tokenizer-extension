#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <working_release_root> <state_root> <mix_root>" >&2
  exit 2
fi

WORKING_RELEASE_ROOT="$1"
STATE_ROOT="$2"
MIX_ROOT="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
LATEST_JSON="${WORKING_RELEASE_ROOT}/dedup_metadata/latest.json"
LATEST_SUCCESS_JSON="${STATE_ROOT}/latest_success.json"
HPLT_INTEGRATION_SUMMARY="${WORKING_RELEASE_ROOT}/hplt_integration_summary.json"
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
  echo "Could not resolve a usable Python interpreter for mix build" >&2
  exit 1
fi
TOKENIZER_REPO_ROOT="${HOME}/Projects/glossapi-tokenizer-extension"
TOKENIZER_REPO_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export GLOSSAPI_WORK_ROOT="${HOME}/data/glossapi_work"

while [ ! -f "${LATEST_JSON}" ] || [ ! -f "${LATEST_SUCCESS_JSON}" ] || [ ! -f "${HPLT_INTEGRATION_SUMMARY}" ]; do
  sleep 60
done

while true; do
  RUN_IDS="$("${PYTHON_BIN}" - "${WORKING_RELEASE_ROOT}" "${STATE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

working_root = Path(sys.argv[1]).resolve()
state_root = Path(sys.argv[2]).resolve()
latest = json.loads((working_root / "dedup_metadata" / "latest.json").read_text(encoding="utf-8"))
latest_success = json.loads((state_root / "latest_success.json").read_text(encoding="utf-8"))
print(latest.get("latest_run_id", ""))
print(latest_success.get("run_id", ""))
PY
)"
  RELEASE_RUN_ID="$(printf '%s\n' "${RUN_IDS}" | sed -n '1p')"
  STATE_RUN_ID="$(printf '%s\n' "${RUN_IDS}" | sed -n '2p')"
  if [ -n "${RELEASE_RUN_ID}" ] && [ "${RELEASE_RUN_ID}" = "${STATE_RUN_ID}" ]; then
    break
  fi
  sleep 60
done

DEDUP_METADATA_ROOT="$("${PYTHON_BIN}" - "${WORKING_RELEASE_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

working_root = Path(sys.argv[1]).resolve()
latest = json.loads((working_root / "dedup_metadata" / "latest.json").read_text(encoding="utf-8"))
print((working_root / latest["builder_metadata_root"]).resolve())
PY
)"

mkdir -p "${MIX_ROOT}/glossapi_only" "${MIX_ROOT}/glossapi_plus_hplt_70_30"

"${PYTHON_BIN}" -m glossapi_corpus_cli.cli mix \
  --output-root "${WORKING_RELEASE_ROOT}" \
  --mix-output-path "${MIX_ROOT}/glossapi_only/mix.parquet" \
  --exclude-needs-ocr-sources openarchives.gr \
  --dedup-metadata-root "${DEDUP_METADATA_ROOT}" \
  --dedup-action drop_intra_and_inter \
  --source-mix-config-path "${TOKENIZER_REPO_ROOT}/subprojects/01_2_training_dataset_mix/examples/glossapi_only_all_non_hplt.json"

"${PYTHON_BIN}" -m glossapi_corpus_cli.cli mix \
  --output-root "${WORKING_RELEASE_ROOT}" \
  --mix-output-path "${MIX_ROOT}/glossapi_plus_hplt_70_30/mix.parquet" \
  --exclude-needs-ocr-sources openarchives.gr \
  --dedup-metadata-root "${DEDUP_METADATA_ROOT}" \
  --dedup-action drop_intra_and_inter \
  --source-mix-config-path "${TOKENIZER_REPO_ROOT}/subprojects/01_2_training_dataset_mix/examples/glossapi_plus_hplt_70_30.json"
