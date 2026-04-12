#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <mix_root> <training_root>" >&2
  exit 2
fi

MIX_ROOT="$1"
TRAINING_ROOT="$2"
GLOSSAPI_MIX="${MIX_ROOT}/glossapi_only/mix.parquet"
MIXED_MIX="${MIX_ROOT}/glossapi_plus_hplt_70_30/mix.parquet"
TRAIN_SCRIPT="${HOME}/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/scripts/train_discovery_tokenizer.py"
VENV_ROOT="${HOME}/venvs/tokenizer-training"

while [ ! -f "${GLOSSAPI_MIX}" ] || [ ! -f "${MIXED_MIX}" ]; do
  sleep 60
done

mkdir -p "${HOME}/venvs" "${TRAINING_ROOT}"
if [ ! -x "${VENV_ROOT}/bin/python" ]; then
  python3 -m venv "${VENV_ROOT}"
fi

"${VENV_ROOT}/bin/python" -m pip install --upgrade pip >/dev/null
"${VENV_ROOT}/bin/python" -m pip install --upgrade transformers tokenizers pyarrow huggingface_hub >/dev/null

systemctl --user stop apertus-discovery-glossapi50k-20260413.service >/dev/null 2>&1 || true
systemctl --user reset-failed apertus-discovery-glossapi50k-20260413.service >/dev/null 2>&1 || true
systemctl --user stop apertus-discovery-mix50k-20260413.service >/dev/null 2>&1 || true
systemctl --user reset-failed apertus-discovery-mix50k-20260413.service >/dev/null 2>&1 || true

systemd-run --user --unit=apertus-discovery-glossapi50k-20260413.service bash -lc \
  "export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=32; \
   ${VENV_ROOT}/bin/python ${TRAIN_SCRIPT} \
     --base-tokenizer swiss-ai/Apertus-8B-2509 \
     --input-glob ${GLOSSAPI_MIX} \
     --output-dir ${TRAINING_ROOT}/glossapi_only_50k \
     --vocab-size 50000 \
     --name glossapi_only_50k \
     >> ${TRAINING_ROOT}/glossapi_only_50k.log 2>&1"

systemd-run --user --unit=apertus-discovery-mix50k-20260413.service bash -lc \
  "export TOKENIZERS_PARALLELISM=true RAYON_NUM_THREADS=32; \
   ${VENV_ROOT}/bin/python ${TRAIN_SCRIPT} \
     --base-tokenizer swiss-ai/Apertus-8B-2509 \
     --input-glob ${MIXED_MIX} \
     --output-dir ${TRAINING_ROOT}/glossapi_plus_hplt_70_30_50k \
     --vocab-size 50000 \
     --name glossapi_plus_hplt_70_30_50k \
     >> ${TRAINING_ROOT}/glossapi_plus_hplt_70_30_50k.log 2>&1"
