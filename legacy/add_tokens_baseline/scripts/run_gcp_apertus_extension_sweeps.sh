#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-eellak-glossapi-20251008}"
ZONE="${ZONE:-europe-west4-b}"
MACHINE_TYPE="${MACHINE_TYPE:-n2-custom-64-262144}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-2000}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-balanced}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2204-lts}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
INSTANCE_NAME="${INSTANCE_NAME:-apertus-greek-tokenizer-${RUN_ID,,}}"

PROJECT_ROOT="/home/foivos/Projects/glossapi-tokenizer-extension"
LOCAL_ARTIFACT_ROOT="${LOCAL_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/${RUN_ID}}"
REMOTE_ROOT="/home/foivos/apertus_greek_tokenizer_runs/${RUN_ID}"
LOCAL_SCRIPT_ROOT="/home/foivos/Projects/glossapi-tokenizer-extension/scripts"
LOCAL_WORK_SCRIPT="/home/foivos/data/glossapi_work/nanochat_glossapi_en_vs_el/scripts/build_apertus_greek_tokenizer_extensions.py"
LOCAL_NANOCHAT_ROOT="/home/foivos/data/glossapi_work/nanochat_glossapi_en_vs_el"
LOCAL_HPLT_MANIFEST="/home/foivos/tmp/hplt-v3-ell-grek-20260329/ell_Grek_manifest.json"
HPLT_DOWNLOAD_JOBS="${HPLT_DOWNLOAD_JOBS:-6}"
HPLT_WORKERS="${HPLT_WORKERS:-6}"
MAX_EVAL_TEXTS_PER_SET="${MAX_EVAL_TEXTS_PER_SET:-256}"
MIN_FREQUENCY="${MIN_FREQUENCY:-5}"
MAX_CANDIDATE_POOL="${MAX_CANDIDATE_POOL:-300000}"
EXTENSION_SIZES="${EXTENSION_SIZES:-10240 15360 20480}"
HPLT_LOCAL_PRUNE_FREQUENCY="${HPLT_LOCAL_PRUNE_FREQUENCY:-2}"

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

wait_for_ssh() {
  local tries=0
  until gcloud compute ssh "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" --command 'echo ready' >/dev/null 2>&1; do
    tries=$((tries + 1))
    if (( tries > 30 )); then
      echo "Instance SSH did not become ready in time." >&2
      exit 1
    fi
    sleep 10
  done
}

require_file "${LOCAL_WORK_SCRIPT}"
require_file "${LOCAL_HPLT_MANIFEST}"
require_file "${LOCAL_NANOCHAT_ROOT}/exports/shard_00000.parquet"
require_file "${LOCAL_NANOCHAT_ROOT}/exports/shard_06542.parquet"
require_file "${LOCAL_NANOCHAT_ROOT}/exports/test.parquet"
require_file "${LOCAL_SCRIPT_ROOT}/gcp_remote_run_apertus_extension_sweeps.sh"

mkdir -p "${LOCAL_ARTIFACT_ROOT}"

gcloud config set project "${PROJECT_ID}" >/dev/null

if ! gcloud compute instances describe "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" >/dev/null 2>&1; then
  gcloud compute instances create "${INSTANCE_NAME}" \
    --project "${PROJECT_ID}" \
    --zone "${ZONE}" \
    --machine-type "${MACHINE_TYPE}" \
    --boot-disk-size "${BOOT_DISK_SIZE_GB}GB" \
    --boot-disk-type "${BOOT_DISK_TYPE}" \
    --image-project "${IMAGE_PROJECT}" \
    --image-family "${IMAGE_FAMILY}" \
    --maintenance-policy MIGRATE \
    --labels "workload=apertus-tokenizer,owner=foivos"
else
  gcloud compute instances start "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" >/dev/null
fi

wait_for_ssh

gcloud compute ssh "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" --command "mkdir -p '${REMOTE_ROOT}/work' '${REMOTE_ROOT}/data/nanochat_glossapi_en_vs_el/exports' '${REMOTE_ROOT}/logs' '${REMOTE_ROOT}/outputs' '${REMOTE_ROOT}/data/hplt_ell_grek'"

gcloud compute scp --project "${PROJECT_ID}" --zone "${ZONE}" \
  "${LOCAL_WORK_SCRIPT}" \
  "${INSTANCE_NAME}:${REMOTE_ROOT}/work/build_apertus_greek_tokenizer_extensions.py"

gcloud compute scp --project "${PROJECT_ID}" --zone "${ZONE}" \
  "${LOCAL_SCRIPT_ROOT}/gcp_remote_run_apertus_extension_sweeps.sh" \
  "${INSTANCE_NAME}:${REMOTE_ROOT}/work/gcp_remote_run_apertus_extension_sweeps.sh"

gcloud compute scp --project "${PROJECT_ID}" --zone "${ZONE}" \
  "${LOCAL_HPLT_MANIFEST}" \
  "${INSTANCE_NAME}:${REMOTE_ROOT}/data/ell_Grek_manifest.json"

gcloud compute scp --project "${PROJECT_ID}" --zone "${ZONE}" \
  "${LOCAL_NANOCHAT_ROOT}/exports/shard_00000.parquet" \
  "${LOCAL_NANOCHAT_ROOT}/exports/shard_06542.parquet" \
  "${LOCAL_NANOCHAT_ROOT}/exports/test.parquet" \
  "${INSTANCE_NAME}:${REMOTE_ROOT}/data/nanochat_glossapi_en_vs_el/exports/"

gcloud compute ssh "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" --command "chmod +x '${REMOTE_ROOT}/work/gcp_remote_run_apertus_extension_sweeps.sh' && HPLT_DOWNLOAD_JOBS='${HPLT_DOWNLOAD_JOBS}' HPLT_WORKERS='${HPLT_WORKERS}' MAX_EVAL_TEXTS_PER_SET='${MAX_EVAL_TEXTS_PER_SET}' MIN_FREQUENCY='${MIN_FREQUENCY}' MAX_CANDIDATE_POOL='${MAX_CANDIDATE_POOL}' EXTENSION_SIZES='${EXTENSION_SIZES}' HPLT_LOCAL_PRUNE_FREQUENCY='${HPLT_LOCAL_PRUNE_FREQUENCY}' '${REMOTE_ROOT}/work/gcp_remote_run_apertus_extension_sweeps.sh' '${REMOTE_ROOT}'"

gcloud compute scp --project "${PROJECT_ID}" --zone "${ZONE}" --recurse \
  "${INSTANCE_NAME}:${REMOTE_ROOT}/outputs" \
  "${INSTANCE_NAME}:${REMOTE_ROOT}/logs" \
  "${LOCAL_ARTIFACT_ROOT}/"

gcloud compute instances stop "${INSTANCE_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" >/dev/null

printf 'Artifacts copied to %s\n' "${LOCAL_ARTIFACT_ROOT}"
