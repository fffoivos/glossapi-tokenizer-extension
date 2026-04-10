#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: gcp_remote_run_apertus_extension_sweeps.sh RUN_ROOT}"

WORK_ROOT="${RUN_ROOT}/work"
DATA_ROOT="${RUN_ROOT}/data"
LOG_ROOT="${RUN_ROOT}/logs"
OUT_ROOT="${RUN_ROOT}/outputs"
VENV_ROOT="${RUN_ROOT}/.venv"
NANOCHAT_ROOT="${DATA_ROOT}/nanochat_glossapi_en_vs_el"
HPLT_ROOT="${DATA_ROOT}/hplt_ell_grek"
SCRIPT_PATH="${WORK_ROOT}/build_apertus_greek_tokenizer_extensions.py"
HPLT_MANIFEST="${DATA_ROOT}/ell_Grek_manifest.json"

HPLT_DOWNLOAD_JOBS="${HPLT_DOWNLOAD_JOBS:-4}"
HPLT_WORKERS="${HPLT_WORKERS:-4}"
MAX_EVAL_TEXTS_PER_SET="${MAX_EVAL_TEXTS_PER_SET:-256}"
MIN_FREQUENCY="${MIN_FREQUENCY:-5}"
MAX_CANDIDATE_POOL="${MAX_CANDIDATE_POOL:-300000}"
EXTENSION_SIZES="${EXTENSION_SIZES:-10240 15360 20480}"
HPLT_LOCAL_PRUNE_FREQUENCY="${HPLT_LOCAL_PRUNE_FREQUENCY:-2}"

mkdir -p "${WORK_ROOT}" "${DATA_ROOT}" "${LOG_ROOT}" "${OUT_ROOT}" "${HPLT_ROOT}"

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip curl jq

python3 -m venv "${VENV_ROOT}"
source "${VENV_ROOT}/bin/activate"
python -m pip install --upgrade pip
python -m pip install "transformers==5.5.0" "pyarrow==23.0.1" "zstandard==0.25.0" "tokenizers==0.22.2"

export HF_HOME="${RUN_ROOT}/hf_home"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

python - <<'PY' "${HPLT_MANIFEST}" > "${WORK_ROOT}/hplt_urls.txt"
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for url in manifest["urls"]:
    print(url)
PY

download_one() {
  local url="$1"
  local out_dir="$2"
  local filename
  filename="$(basename "${url}")"
  if [[ -s "${out_dir}/${filename}" ]]; then
    return 0
  fi
  curl -L --fail --retry 8 --retry-delay 10 --continue-at - --output "${out_dir}/${filename}.part" "${url}"
  mv "${out_dir}/${filename}.part" "${out_dir}/${filename}"
}

export -f download_one

(
  xargs -a "${WORK_ROOT}/hplt_urls.txt" -n 1 -P "${HPLT_DOWNLOAD_JOBS}" -I '{}' bash -lc 'download_one "$1" "$2"' _ '{}' "${HPLT_ROOT}"
) > "${LOG_ROOT}/hplt_download.log" 2>&1 &
HPLT_DOWNLOAD_PID="$!"

python "${SCRIPT_PATH}" \
  --mode nanochat_only \
  --output-dir "${OUT_ROOT}/nanochat_only" \
  --nanochat-root "${NANOCHAT_ROOT}" \
  --extension-sizes ${EXTENSION_SIZES} \
  --min-frequency "${MIN_FREQUENCY}" \
  --max-candidate-pool "${MAX_CANDIDATE_POOL}" \
  --max-eval-texts-per-set "${MAX_EVAL_TEXTS_PER_SET}" \
  > "${LOG_ROOT}/nanochat_only.log" 2>&1

wait "${HPLT_DOWNLOAD_PID}"

mapfile -t HPLT_FILES < <(find "${HPLT_ROOT}" -maxdepth 1 -type f -name '*.jsonl.zst' | sort)
if [[ "${#HPLT_FILES[@]}" -eq 0 ]]; then
  echo "No HPLT shard files were downloaded into ${HPLT_ROOT}" >&2
  exit 1
fi

HPLT_INPUT_ARGS=()
for path in "${HPLT_FILES[@]}"; do
  HPLT_INPUT_ARGS+=(--hplt-input "${path}")
done

python "${SCRIPT_PATH}" \
  --mode nanochat_plus_hplt \
  --output-dir "${OUT_ROOT}/nanochat_plus_hplt" \
  --nanochat-root "${NANOCHAT_ROOT}" \
  --extension-sizes ${EXTENSION_SIZES} \
  --min-frequency "${MIN_FREQUENCY}" \
  --max-candidate-pool "${MAX_CANDIDATE_POOL}" \
  --max-eval-texts-per-set "${MAX_EVAL_TEXTS_PER_SET}" \
  --hplt-workers "${HPLT_WORKERS}" \
  --hplt-local-prune-frequency "${HPLT_LOCAL_PRUNE_FREQUENCY}" \
  "${HPLT_INPUT_ARGS[@]}" \
  > "${LOG_ROOT}/nanochat_plus_hplt.log" 2>&1

printf '%s\n' "${OUT_ROOT}/nanochat_only" > "${OUT_ROOT}/nanochat_only_path.txt"
printf '%s\n' "${OUT_ROOT}/nanochat_plus_hplt" > "${OUT_ROOT}/nanochat_plus_hplt_path.txt"
