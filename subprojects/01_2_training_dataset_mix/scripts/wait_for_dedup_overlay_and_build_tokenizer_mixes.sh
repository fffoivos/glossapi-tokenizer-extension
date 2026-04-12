#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <working_release_root> <state_root> <mix_root>" >&2
  exit 2
fi

WORKING_RELEASE_ROOT="$1"
STATE_ROOT="$2"
MIX_ROOT="$3"
LATEST_JSON="${WORKING_RELEASE_ROOT}/dedup_metadata/latest.json"
LATEST_SUCCESS_JSON="${STATE_ROOT}/latest_success.json"
HPLT_INTEGRATION_SUMMARY="${WORKING_RELEASE_ROOT}/hplt_integration_summary.json"
PYTHON_BIN="${HOME}/venvs/glossapi-corpus-clean/bin/python"
TOKENIZER_REPO_ROOT="${HOME}/Projects/glossapi-tokenizer-extension"
export PYTHONPATH="${HOME}/data/glossapi_work"
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
