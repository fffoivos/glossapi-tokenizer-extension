#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <mix_root> <training_root>" >&2
  exit 2
fi

MIX_ROOT="$1"
TRAINING_ROOT="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../" && pwd)"
GLOSSAPI_MIX="${MIX_ROOT}/glossapi_only/mix.parquet"
MIXED_MIX="${MIX_ROOT}/glossapi_plus_hplt_70_30/mix.parquet"
TRAIN_SCRIPT="${REPO_ROOT}/subprojects/02_1_tokenizer_experiments/scripts/train_discovery_tokenizer.py"
VENV_ROOT="${TOKENIZER_TRAINING_VENV_ROOT:-${HOME}/venvs/tokenizer-training}"
PYTHON_BIN="${TOKENIZER_TRAINING_PYTHON_BIN:-${VENV_ROOT}/bin/python}"
BASE_TOKENIZER="${TOKENIZER_TRAINING_BASE_TOKENIZER:-swiss-ai/Apertus-8B-2509}"
VOCAB_SIZE="${TOKENIZER_TRAINING_VOCAB_SIZE:-50000}"
RAYON_THREADS="${TOKENIZER_TRAINING_RAYON_THREADS:-32}"
GLOSSAPI_NAME="${TOKENIZER_TRAINING_GLOSSAPI_NAME:-glossapi_only_50k}"
MIXED_NAME="${TOKENIZER_TRAINING_MIXED_NAME:-glossapi_plus_hplt_70_30_50k}"
LAUNCH_MODE="${TOKENIZER_TRAINING_LAUNCH_MODE:-systemd}"
INSTALL_DEPS="${TOKENIZER_TRAINING_INSTALL_DEPS:-1}"
WAIT_INTERVAL_SECONDS="${TOKENIZER_TRAINING_WAIT_INTERVAL_SECONDS:-15}"

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

write_training_status() {
  local status_path="$1"
  local run_name="$2"
  local state="$3"
  local input_glob="$4"
  local output_dir="$5"
  local log_path="$6"
  local unit_name="$7"
  local pid="$8"
  local exit_code="$9"
  local note="${10}"

  STATUS_PATH="${status_path}" \
  RUN_NAME="${run_name}" \
  STATE="${state}" \
  INPUT_GLOB="${input_glob}" \
  OUTPUT_DIR="${output_dir}" \
  LOG_PATH="${log_path}" \
  UNIT_NAME="${unit_name}" \
  PID_VALUE="${pid}" \
  EXIT_CODE_VALUE="${exit_code}" \
  NOTE_VALUE="${note}" \
  TIMESTAMP_UTC="$(timestamp_utc)" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

status_path = Path(os.environ["STATUS_PATH"])
status_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "run_name": os.environ["RUN_NAME"],
    "state": os.environ["STATE"],
    "updated_at_utc": os.environ["TIMESTAMP_UTC"],
    "input_glob": os.environ["INPUT_GLOB"],
    "output_dir": os.environ["OUTPUT_DIR"],
    "log_path": os.environ["LOG_PATH"],
}
unit_name = os.environ["UNIT_NAME"]
if unit_name:
    payload["systemd_unit"] = unit_name
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

mkdir -p "${HOME}/venvs" "${TRAINING_ROOT}"
if [ ! -x "${PYTHON_BIN}" ]; then
  python3 -m venv "${VENV_ROOT}"
fi

if [ "${INSTALL_DEPS}" != "0" ]; then
  "${PYTHON_BIN}" -m pip install --upgrade pip >/dev/null
  "${PYTHON_BIN}" -m pip install --upgrade transformers tokenizers pyarrow huggingface_hub >/dev/null
fi

run_training_inline() {
  local input_glob="$1"
  local output_dir="$2"
  local run_name="$3"
  local log_path="$4"
  export TOKENIZERS_PARALLELISM=true
  export RAYON_NUM_THREADS="${RAYON_THREADS}"
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --base-tokenizer "${BASE_TOKENIZER}" \
    --input-glob "${input_glob}" \
    --output-dir "${output_dir}" \
    --vocab-size "${VOCAB_SIZE}" \
    --name "${run_name}" \
    >> "${log_path}" 2>&1
}

ensure_deps_installed() {
  if [ "${INSTALL_DEPS}" != "0" ]; then
    "${PYTHON_BIN}" -m pip install --upgrade pip >/dev/null
    "${PYTHON_BIN}" -m pip install --upgrade transformers tokenizers pyarrow huggingface_hub >/dev/null
  fi
}

declare -a TARGET_NAMES=("${GLOSSAPI_NAME}" "${MIXED_NAME}")
declare -a TARGET_INPUTS=("${GLOSSAPI_MIX}" "${MIXED_MIX}")
declare -a TARGET_UNITS=("apertus-discovery-glossapi50k-20260413.service" "apertus-discovery-mix50k-20260413.service")
declare -a TARGET_MIX_STATUS=("${MIX_ROOT}/glossapi_only/build_status.json" "${MIX_ROOT}/glossapi_plus_hplt_70_30/build_status.json")

declare -a TARGET_OUTPUTS=()
declare -a TARGET_LOGS=()
declare -a TARGET_STATUS=()
declare -a TARGET_STARTED=()
declare -a TARGET_PIDS=()
declare -a TARGET_COMPLETED=()

for run_name in "${TARGET_NAMES[@]}"; do
  TARGET_OUTPUTS+=("${TRAINING_ROOT}/${run_name}")
  TARGET_LOGS+=("${TRAINING_ROOT}/${run_name}.log")
  TARGET_STATUS+=("${TRAINING_ROOT}/${run_name}.launch_status.json")
  TARGET_STARTED+=("0")
  TARGET_PIDS+=("")
  TARGET_COMPLETED+=("0")
done

if [ "${LAUNCH_MODE}" != "inline" ]; then
  systemctl --user stop apertus-discovery-glossapi50k-20260413.service >/dev/null 2>&1 || true
  systemctl --user reset-failed apertus-discovery-glossapi50k-20260413.service >/dev/null 2>&1 || true
  systemctl --user stop apertus-discovery-mix50k-20260413.service >/dev/null 2>&1 || true
  systemctl --user reset-failed apertus-discovery-mix50k-20260413.service >/dev/null 2>&1 || true
fi

deps_ready=0
targets_remaining=${#TARGET_NAMES[@]}

mix_is_ready() {
  local input_glob="$1"
  local mix_status_path="$2"
  python3 - "$input_glob" "$mix_status_path" <<'PY'
import json
import sys
from pathlib import Path

mix_path = Path(sys.argv[1])
status_path = Path(sys.argv[2])

if not mix_path.exists() or mix_path.stat().st_size <= 0:
    raise SystemExit(1)
if not status_path.exists():
    raise SystemExit(1)
try:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if payload.get("state") != "completed":
    raise SystemExit(1)
status_mix_output = payload.get("mix_output_path")
if status_mix_output and Path(status_mix_output).resolve() != mix_path.resolve():
    raise SystemExit(1)
raise SystemExit(0)
PY
}

launch_target() {
  local idx="$1"
  local input_glob="${TARGET_INPUTS[$idx]}"
  local output_dir="${TARGET_OUTPUTS[$idx]}"
  local run_name="${TARGET_NAMES[$idx]}"
  local log_path="${TARGET_LOGS[$idx]}"
  local status_path="${TARGET_STATUS[$idx]}"
  local unit_name="${TARGET_UNITS[$idx]}"

  mkdir -p "${output_dir}"
  if [ -f "${output_dir}/training_summary.json" ]; then
    TARGET_STARTED[$idx]="1"
    TARGET_COMPLETED[$idx]="1"
    targets_remaining=$((targets_remaining - 1))
    write_training_status "${status_path}" "${run_name}" "completed" "${input_glob}" "${output_dir}" "${log_path}" "${unit_name}" "" "" "skipped_existing_summary"
    return 0
  fi
  if [ "${deps_ready}" = "0" ]; then
    ensure_deps_installed
    deps_ready=1
  fi
  : > "${log_path}"
  if [ "${LAUNCH_MODE}" = "inline" ]; then
    write_training_status "${status_path}" "${run_name}" "launching" "${input_glob}" "${output_dir}" "${log_path}" "${unit_name}" "" "" ""
    (
      run_training_inline "${input_glob}" "${output_dir}" "${run_name}" "${log_path}"
    ) &
    local pid="$!"
    TARGET_PIDS[$idx]="${pid}"
    write_training_status "${status_path}" "${run_name}" "running" "${input_glob}" "${output_dir}" "${log_path}" "${unit_name}" "${pid}" "" ""
    targets_remaining=$((targets_remaining - 1))
  else
    write_training_status "${status_path}" "${run_name}" "launching" "${input_glob}" "${output_dir}" "${log_path}" "${unit_name}" "" "" ""
    systemd-run --user --unit="${unit_name}" bash -lc \
      "export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=${RAYON_THREADS}; \
       ${PYTHON_BIN} ${TRAIN_SCRIPT} \
         --base-tokenizer ${BASE_TOKENIZER} \
         --input-glob ${input_glob} \
         --output-dir ${output_dir} \
         --vocab-size ${VOCAB_SIZE} \
         --name ${run_name} \
         >> ${log_path} 2>&1" >/dev/null
    write_training_status "${status_path}" "${run_name}" "launched_systemd" "${input_glob}" "${output_dir}" "${log_path}" "${unit_name}" "" "" ""
    TARGET_COMPLETED[$idx]="1"
    targets_remaining=$((targets_remaining - 1))
  fi
  TARGET_STARTED[$idx]="1"
}

while [ "${targets_remaining}" -gt 0 ]; do
  idx=0
  launched_this_round=0
  for input_glob in "${TARGET_INPUTS[@]}"; do
    if [ "${TARGET_STARTED[$idx]}" = "1" ]; then
      idx=$((idx + 1))
      continue
    fi
    write_training_status "${TARGET_STATUS[$idx]}" "${TARGET_NAMES[$idx]}" "waiting_for_mix" "${TARGET_INPUTS[$idx]}" "${TARGET_OUTPUTS[$idx]}" "${TARGET_LOGS[$idx]}" "${TARGET_UNITS[$idx]}" "" "" ""
    if mix_is_ready "${input_glob}" "${TARGET_MIX_STATUS[$idx]}"; then
      launch_target "${idx}"
      launched_this_round=1
    fi
    idx=$((idx + 1))
  done
  if [ "${targets_remaining}" -gt 0 ] && [ "${launched_this_round}" = "0" ]; then
    sleep "${WAIT_INTERVAL_SECONDS}"
  fi
done

if [ "${LAUNCH_MODE}" = "inline" ]; then
  idx=0
  failed=0
  for pid in "${TARGET_PIDS[@]}"; do
    if [ -z "${pid}" ]; then
      idx=$((idx + 1))
      continue
    fi
    if wait "${pid}"; then
      write_training_status "${TARGET_STATUS[$idx]}" "${TARGET_NAMES[$idx]}" "completed" "${TARGET_INPUTS[$idx]}" "${TARGET_OUTPUTS[$idx]}" "${TARGET_LOGS[$idx]}" "${TARGET_UNITS[$idx]}" "${pid}" "0" ""
    else
      exit_code="$?"
      write_training_status "${TARGET_STATUS[$idx]}" "${TARGET_NAMES[$idx]}" "failed" "${TARGET_INPUTS[$idx]}" "${TARGET_OUTPUTS[$idx]}" "${TARGET_LOGS[$idx]}" "${TARGET_UNITS[$idx]}" "${pid}" "${exit_code}" "see_log"
      failed=1
    fi
    idx=$((idx + 1))
  done
  exit "${failed}"
fi
