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
MIX_MAX_JOBS="${TOKENIZER_MIX_MAX_JOBS:-2}"
SHARED_ROOT="${MIX_ROOT}/_shared"
SELECTED_INPUT_PATH="${SHARED_ROOT}/selected_input.parquet"
SELECTED_INPUT_STATUS="${SHARED_ROOT}/prepare_status.json"
SELECTED_INPUT_LOG="${SHARED_ROOT}/prepare.log"
SELECTED_INPUT_SUMMARY="${SHARED_ROOT}/selected_input_summary.json"

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_mix_status() {
  local status_path="$1"
  local mix_name="$2"
  local state="$3"
  local mix_output_path="$4"
  local build_log_path="$5"
  local config_path="$6"
  local pid="$7"
  local exit_code="$8"
  local note="$9"

  STATUS_PATH="${status_path}" \
  MIX_NAME="${mix_name}" \
  STATE="${state}" \
  MIX_OUTPUT_PATH="${mix_output_path}" \
  BUILD_LOG_PATH="${build_log_path}" \
  CONFIG_PATH="${config_path}" \
  PID_VALUE="${pid}" \
  EXIT_CODE_VALUE="${exit_code}" \
  NOTE_VALUE="${note}" \
  TIMESTAMP_UTC="$(timestamp_utc)" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

status_path = Path(os.environ["STATUS_PATH"])
status_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "mix_name": os.environ["MIX_NAME"],
    "state": os.environ["STATE"],
    "updated_at_utc": os.environ["TIMESTAMP_UTC"],
    "mix_output_path": os.environ["MIX_OUTPUT_PATH"],
    "build_log_path": os.environ["BUILD_LOG_PATH"],
    "source_mix_config_path": os.environ["CONFIG_PATH"],
}
pid_value = os.environ["PID_VALUE"]
if pid_value:
    payload["pid"] = int(pid_value)
exit_code_value = os.environ["EXIT_CODE_VALUE"]
if exit_code_value:
    payload["exit_code"] = int(exit_code_value)
note_value = os.environ["NOTE_VALUE"]
if note_value:
    payload["note"] = note_value
status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

write_selected_input_status() {
  local state="$1"
  local note="$2"

  STATUS_PATH="${SELECTED_INPUT_STATUS}" \
  STATE="${state}" \
  NOTE_VALUE="${note}" \
  SELECTED_INPUT_PATH_VALUE="${SELECTED_INPUT_PATH}" \
  BUILD_LOG_PATH="${SELECTED_INPUT_LOG}" \
  SUMMARY_PATH="${SELECTED_INPUT_SUMMARY}" \
  TIMESTAMP_UTC="$(timestamp_utc)" \
  "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

status_path = Path(os.environ["STATUS_PATH"])
status_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "state": os.environ["STATE"],
    "updated_at_utc": os.environ["TIMESTAMP_UTC"],
    "selected_input_path": os.environ["SELECTED_INPUT_PATH_VALUE"],
    "build_log_path": os.environ["BUILD_LOG_PATH"],
    "summary_path": os.environ["SUMMARY_PATH"],
}
note_value = os.environ["NOTE_VALUE"]
if note_value:
    payload["note"] = note_value
status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

prepare_selected_input() {
  mkdir -p "${SHARED_ROOT}"
  if [ -s "${SELECTED_INPUT_PATH}" ]; then
    write_selected_input_status "completed" "skipped_existing_selected_input"
    return 0
  fi
  rm -f "${SELECTED_INPUT_PATH}" "${SELECTED_INPUT_SUMMARY}"
  : > "${SELECTED_INPUT_LOG}"
  write_selected_input_status "running" ""
  if "${PYTHON_BIN}" -m glossapi_corpus_cli.cli mix-prepare-selected-input \
      --output-root "${WORKING_RELEASE_ROOT}" \
      --selected-input-path "${SELECTED_INPUT_PATH}" \
      --exclude-needs-ocr-sources openarchives.gr \
      --dedup-metadata-root "${DEDUP_METADATA_ROOT}" \
      --dedup-action drop_intra_and_inter \
      > "${SELECTED_INPUT_SUMMARY}" 2>> "${SELECTED_INPUT_LOG}"; then
    write_selected_input_status "completed" ""
  else
    write_selected_input_status "failed" "see_prepare_log"
    return 1
  fi
}

declare -a MIX_NAMES=()
declare -a MIX_PIDS=()
declare -a MIX_STATUS_PATHS=()
declare -a MIX_OUTPUT_PATHS=()
declare -a MIX_LOG_PATHS=()
declare -a MIX_CONFIG_PATHS=()

launch_mix_build() {
  local mix_name="$1"
  local config_path="$2"
  local mix_dir="${MIX_ROOT}/${mix_name}"
  local mix_output_path="${mix_dir}/mix.parquet"
  local build_log_path="${mix_dir}/build.log"
  local status_path="${mix_dir}/build_status.json"

  mkdir -p "${mix_dir}"
  if [ -s "${mix_output_path}" ]; then
    write_mix_status "${status_path}" "${mix_name}" "completed" "${mix_output_path}" "${build_log_path}" "${config_path}" "" "" "skipped_existing_output"
    return 0
  fi

  : > "${build_log_path}"
  write_mix_status "${status_path}" "${mix_name}" "launching" "${mix_output_path}" "${build_log_path}" "${config_path}" "" "" ""
  (
    exec "${PYTHON_BIN}" -m glossapi_corpus_cli.cli mix-build-from-selected-input \
      --selected-input-path "${SELECTED_INPUT_PATH}" \
      --mix-output-path "${mix_output_path}" \
      --source-mix-config-path "${config_path}"
  ) >> "${build_log_path}" 2>&1 &
  local pid="$!"
  write_mix_status "${status_path}" "${mix_name}" "running" "${mix_output_path}" "${build_log_path}" "${config_path}" "${pid}" "" ""
  MIX_NAMES+=("${mix_name}")
  MIX_PIDS+=("${pid}")
  MIX_STATUS_PATHS+=("${status_path}")
  MIX_OUTPUT_PATHS+=("${mix_output_path}")
  MIX_LOG_PATHS+=("${build_log_path}")
  MIX_CONFIG_PATHS+=("${config_path}")
}

wait_mix_builds() {
  local idx
  local failed=0
  for idx in "${!MIX_PIDS[@]}"; do
    local pid="${MIX_PIDS[$idx]}"
    local mix_name="${MIX_NAMES[$idx]}"
    local status_path="${MIX_STATUS_PATHS[$idx]}"
    local mix_output_path="${MIX_OUTPUT_PATHS[$idx]}"
    local build_log_path="${MIX_LOG_PATHS[$idx]}"
    local config_path="${MIX_CONFIG_PATHS[$idx]}"
    local exit_code=0
    if wait "${pid}"; then
      write_mix_status "${status_path}" "${mix_name}" "completed" "${mix_output_path}" "${build_log_path}" "${config_path}" "${pid}" "0" ""
    else
      exit_code="$?"
      write_mix_status "${status_path}" "${mix_name}" "failed" "${mix_output_path}" "${build_log_path}" "${config_path}" "${pid}" "${exit_code}" "see_build_log"
      failed=1
    fi
  done
  MIX_NAMES=()
  MIX_PIDS=()
  MIX_STATUS_PATHS=()
  MIX_OUTPUT_PATHS=()
  MIX_LOG_PATHS=()
  MIX_CONFIG_PATHS=()
  return "${failed}"
}

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

mkdir -p "${MIX_ROOT}/glossapi_only" "${MIX_ROOT}/glossapi_plus_hplt_70_30" "${SHARED_ROOT}"

GLOSSAPI_ONLY_CONFIG="${TOKENIZER_REPO_ROOT}/subprojects/01_2_training_dataset_mix/examples/glossapi_only_all_non_hplt.json"
GLOSSAPI_PLUS_HPLT_CONFIG="${TOKENIZER_REPO_ROOT}/subprojects/01_2_training_dataset_mix/examples/glossapi_plus_hplt_70_30.json"

prepare_selected_input

if [ "${MIX_MAX_JOBS}" -le 1 ]; then
  launch_mix_build "glossapi_only" "${GLOSSAPI_ONLY_CONFIG}"
  wait_mix_builds
  launch_mix_build "glossapi_plus_hplt_70_30" "${GLOSSAPI_PLUS_HPLT_CONFIG}"
  wait_mix_builds
else
  launch_mix_build "glossapi_only" "${GLOSSAPI_ONLY_CONFIG}"
  launch_mix_build "glossapi_plus_hplt_70_30" "${GLOSSAPI_PLUS_HPLT_CONFIG}"
  wait_mix_builds
fi
