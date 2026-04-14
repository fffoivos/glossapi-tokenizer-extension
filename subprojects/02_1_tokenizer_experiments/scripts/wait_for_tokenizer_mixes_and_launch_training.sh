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

while [ ! -f "${GLOSSAPI_MIX}" ] || [ ! -f "${MIXED_MIX}" ]; do
  sleep 60
done

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

if [ "${LAUNCH_MODE}" = "inline" ]; then
  run_training_inline \
    "${GLOSSAPI_MIX}" \
    "${TRAINING_ROOT}/${GLOSSAPI_NAME}" \
    "${GLOSSAPI_NAME}" \
    "${TRAINING_ROOT}/${GLOSSAPI_NAME}.log"
  run_training_inline \
    "${MIXED_MIX}" \
    "${TRAINING_ROOT}/${MIXED_NAME}" \
    "${MIXED_NAME}" \
    "${TRAINING_ROOT}/${MIXED_NAME}.log"
  exit 0
fi

systemctl --user stop apertus-discovery-glossapi50k-20260413.service >/dev/null 2>&1 || true
systemctl --user reset-failed apertus-discovery-glossapi50k-20260413.service >/dev/null 2>&1 || true
systemctl --user stop apertus-discovery-mix50k-20260413.service >/dev/null 2>&1 || true
systemctl --user reset-failed apertus-discovery-mix50k-20260413.service >/dev/null 2>&1 || true

systemd-run --user --unit=apertus-discovery-glossapi50k-20260413.service bash -lc \
  "export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=${RAYON_THREADS}; \
   ${PYTHON_BIN} ${TRAIN_SCRIPT} \
     --base-tokenizer ${BASE_TOKENIZER} \
     --input-glob ${GLOSSAPI_MIX} \
     --output-dir ${TRAINING_ROOT}/${GLOSSAPI_NAME} \
     --vocab-size ${VOCAB_SIZE} \
     --name ${GLOSSAPI_NAME} \
     >> ${TRAINING_ROOT}/${GLOSSAPI_NAME}.log 2>&1"

systemd-run --user --unit=apertus-discovery-mix50k-20260413.service bash -lc \
  "export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=${RAYON_THREADS}; \
   ${PYTHON_BIN} ${TRAIN_SCRIPT} \
     --base-tokenizer ${BASE_TOKENIZER} \
     --input-glob ${MIXED_MIX} \
     --output-dir ${TRAINING_ROOT}/${MIXED_NAME} \
     --vocab-size ${VOCAB_SIZE} \
     --name ${MIXED_NAME} \
     >> ${TRAINING_ROOT}/${MIXED_NAME}.log 2>&1"
