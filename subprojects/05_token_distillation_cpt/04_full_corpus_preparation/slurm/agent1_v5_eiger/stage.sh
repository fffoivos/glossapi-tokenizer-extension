#!/usr/bin/env bash
set -euo pipefail

: "${STAGE:?STAGE is required}"
: "${PIPELINE_ROOT:?PIPELINE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

CONFIG="${CONFIG:-${PIPELINE_ROOT}/configs/agent1_v5_eiger_pipeline.json}"
PYTHON="${VENV_ROOT:-${RUN_ROOT}/runtime/venv}/bin/python"
RUNTIME_ROOT="${RUNTIME_ROOT:-${RUN_ROOT}.runtime}"
PIPELINE="${PIPELINE_ROOT}/scripts/agent1_v5_pipeline.py"
DEDUP="${PIPELINE_ROOT}/scripts/agent1_v5_datatrove.py"
PUBLISH="${PIPELINE_ROOT}/scripts/publish_private_agent1_v5.py"
CONTRACT="${RUN_ROOT}/run_contract.json"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

case "${STAGE}" in
  setup)
    : "${GLOSSAPI_ROOT:?GLOSSAPI_ROOT is required}"
    : "${DATATROVE_ROOT:?DATATROVE_ROOT is required}"
    GLOSSAPI_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pins"]["glossapi_commit"])' "${CONFIG}")"
    DATATROVE_COMMIT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pins"]["datatrove_commit"])' "${CONFIG}")"
    RUST_TOOLCHAIN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pins"]["rust_toolchain"])' "${CONFIG}")"
    if [[ ! -d "${GLOSSAPI_ROOT}/.git" ]]; then
      git clone https://github.com/eellak/glossAPI.git "${GLOSSAPI_ROOT}"
    fi
    if [[ ! -d "${DATATROVE_ROOT}/.git" ]]; then
      git clone https://github.com/huggingface/datatrove.git "${DATATROVE_ROOT}"
    fi
    git -C "${GLOSSAPI_ROOT}" fetch --quiet origin "${GLOSSAPI_COMMIT}"
    git -C "${GLOSSAPI_ROOT}" checkout --detach "${GLOSSAPI_COMMIT}"
    git -C "${DATATROVE_ROOT}" fetch --quiet origin "${DATATROVE_COMMIT}"
    git -C "${DATATROVE_ROOT}" checkout --detach "${DATATROVE_COMMIT}"
    test -z "$(git -C "${GLOSSAPI_ROOT}" status --porcelain)"
    test -z "$(git -C "${DATATROVE_ROOT}" status --porcelain)"
    mkdir -p "${RUNTIME_ROOT}"
    export CARGO_HOME="${RUNTIME_ROOT}/cargo"
    export RUSTUP_HOME="${RUNTIME_ROOT}/rustup"
    export PATH="${CARGO_HOME}/bin:${PATH}"
    if [[ ! -x "${CARGO_HOME}/bin/cargo" ]]; then
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --no-modify-path --profile minimal --default-toolchain "${RUST_TOOLCHAIN}"
    fi
    test "$(rustc --version | awk '{print $2}')" = "${RUST_TOOLCHAIN}"
    python3 -m venv "${VENV_ROOT}"
    "${PYTHON}" -m pip install --upgrade pip
    "${PYTHON}" -m pip install -r "${PIPELINE_ROOT}/configs/agent1_v5_requirements.txt"
    "${PYTHON}" -m pip install --no-deps -e "${DATATROVE_ROOT}"
    "${PYTHON}" -m pip install --no-deps -e "${GLOSSAPI_ROOT}"
    "${PYTHON}" -m pip install -e "${GLOSSAPI_ROOT}/rust/glossapi_rs_cleaner"
    "${PYTHON}" -m pip install -e "${GLOSSAPI_ROOT}/rust/glossapi_rs_noise"
    "${PYTHON}" -m pip check
    "${PYTHON}" -m pip freeze --all > "${RUNTIME_ROOT}/pip-freeze.txt.partial"
    mv "${RUNTIME_ROOT}/pip-freeze.txt.partial" "${RUNTIME_ROOT}/pip-freeze.txt"
    ;;
  bootstrap)
    : "${ACQUISITION_RECEIPT:?ACQUISITION_RECEIPT is required}"
    : "${GLOSSAPI_ROOT:?GLOSSAPI_ROOT is required}"
    : "${DATATROVE_ROOT:?DATATROVE_ROOT is required}"
    : "${RUN_ID:?RUN_ID is required}"
    "${PYTHON}" "${PIPELINE}" freeze-contract \
      --config "${CONFIG}" \
      --acquisition-receipt "${ACQUISITION_RECEIPT}" \
      --glossapi-root "${GLOSSAPI_ROOT}" \
      --run-root "${RUN_ROOT}" \
      --run-id "${RUN_ID}"
    "${PYTHON}" "${PIPELINE}" plan-transform --config "${CONFIG}" --contract "${CONTRACT}" --output "${RUN_ROOT}/transform_tasks.json"
    "${PYTHON}" "${PIPELINE}" plan-base --contract "${CONTRACT}" --output "${RUN_ROOT}/base_tasks.json"
    "${PYTHON}" "${PIPELINE}" build-glossapi-runtime-receipt --config "${CONFIG}" --glossapi-root "${GLOSSAPI_ROOT}" --output "${RUN_ROOT}/glossapi_runtime.json"
    "${PYTHON}" "${DEDUP}" build-runtime-receipt --config "${CONFIG}" --datatrove-root "${DATATROVE_ROOT}" --output "${RUN_ROOT}/datatrove_runtime.json"
    ;;
  transform)
    "${PYTHON}" "${PIPELINE}" transform-task --config "${CONFIG}" --contract "${CONTRACT}" --tasks "${RUN_ROOT}/transform_tasks.json" --task-index "${TASK_INDEX}"
    ;;
  merge-transform)
    "${PYTHON}" "${PIPELINE}" merge-transform --config "${CONFIG}" --contract "${CONTRACT}" --tasks "${RUN_ROOT}/transform_tasks.json" --output "${RUN_ROOT}/transform_manifest.json"
    ;;
  glossapi)
    "${PYTHON}" "${PIPELINE}" glossapi-task --config "${CONFIG}" --contract "${CONTRACT}" --transform-manifest "${RUN_ROOT}/transform_manifest.json" --runtime-receipt "${RUN_ROOT}/glossapi_runtime.json" --task-index "${TASK_INDEX}" --scratch-root "${SCRATCH_ROOT:-${RUN_ROOT}/task-scratch}" --threads "${GLOSSAPI_THREADS:-8}"
    ;;
  merge-glossapi)
    "${PYTHON}" "${PIPELINE}" merge-glossapi --contract "${CONTRACT}" --transform-manifest "${RUN_ROOT}/transform_manifest.json" --output "${RUN_ROOT}/glossapi_manifest.json"
    ;;
  plan-envelope)
    "${PYTHON}" "${PIPELINE}" plan-envelope --contract "${CONTRACT}" --glossapi-manifest "${RUN_ROOT}/glossapi_manifest.json" --output "${RUN_ROOT}/envelope_plan.json"
    ;;
  envelope)
    "${PYTHON}" "${PIPELINE}" envelope-task --contract "${CONTRACT}" --envelope-plan "${RUN_ROOT}/envelope_plan.json" --task-index "${TASK_INDEX}"
    ;;
  merge-envelope)
    "${PYTHON}" "${PIPELINE}" merge-envelope --contract "${CONTRACT}" --envelope-plan "${RUN_ROOT}/envelope_plan.json" --output "${RUN_ROOT}/candidate_manifest.json"
    ;;
  base)
    "${PYTHON}" "${PIPELINE}" cast-base-task --contract "${CONTRACT}" --base-plan "${RUN_ROOT}/base_tasks.json" --task-index "${TASK_INDEX}"
    ;;
  merge-base)
    "${PYTHON}" "${PIPELINE}" merge-base --contract "${CONTRACT}" --base-plan "${RUN_ROOT}/base_tasks.json" --output "${RUN_ROOT}/base_manifest.json"
    ;;
  combine)
    "${PYTHON}" "${PIPELINE}" combine-release --contract "${CONTRACT}" --base-manifest "${RUN_ROOT}/base_manifest.json" --candidate-manifest "${RUN_ROOT}/candidate_manifest.json" --output "${RUN_ROOT}/release-pre-dedup"
    ;;
  publish-pre)
    : "${HF_TOKEN_FILE:?HF_TOKEN_FILE is required for publication}"
    test "$(stat -c %a "${HF_TOKEN_FILE}")" = "600" || { echo "HF token file must have mode 600" >&2; exit 2; }
    HF_TOKEN="$(<"${HF_TOKEN_FILE}")" "${PYTHON}" "${PUBLISH}" --release "${RUN_ROOT}/release-pre-dedup" --kind pre-dedup --repo-id "$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["private_repositories"]["pre_dedup"])' "${CONTRACT}")" --staging "${RUN_ROOT}/publication-staging/pre-dedup" --output "${RUN_ROOT}/publication-pre.json" --execute
    ;;
  exact-index)
    "${PYTHON}" "${DEDUP}" exact-index-task --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --task-index "${TASK_INDEX}"
    ;;
  merge-exact)
    "${PYTHON}" "${DEDUP}" merge-exact-index --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --output "${RUN_ROOT}/exact_manifest.json"
    ;;
  signature)
    "${PYTHON}" "${DEDUP}" signature-task --config "${CONFIG}" --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --runtime-receipt "${RUN_ROOT}/datatrove_runtime.json" --task-index "${TASK_INDEX}"
    ;;
  merge-signatures)
    "${PYTHON}" "${DEDUP}" merge-signatures --config "${CONFIG}" --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --output "${RUN_ROOT}/signature_manifest.json"
    ;;
  bucket)
    "${PYTHON}" "${DEDUP}" bucket-task --config "${CONFIG}" --contract "${CONTRACT}" --signature-manifest "${RUN_ROOT}/signature_manifest.json" --runtime-receipt "${RUN_ROOT}/datatrove_runtime.json" --task-index "${TASK_INDEX}"
    ;;
  merge-pairs)
    "${PYTHON}" "${DEDUP}" merge-lsh-pairs --config "${CONFIG}" --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --output "${RUN_ROOT}/lsh_pairs.sqlite"
    ;;
  shingles)
    "${PYTHON}" "${DEDUP}" shingle-task --config "${CONFIG}" --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --pair-database "${RUN_ROOT}/lsh_pairs.sqlite" --runtime-receipt "${RUN_ROOT}/datatrove_runtime.json" --task-index "${TASK_INDEX}"
    ;;
  merge-shingles)
    "${PYTHON}" "${DEDUP}" merge-shingles --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --pair-database "${RUN_ROOT}/lsh_pairs.sqlite" --output "${RUN_ROOT}/shingle_manifest.json"
    ;;
  verify)
    "${PYTHON}" "${DEDUP}" verify-task --config "${CONFIG}" --contract "${CONTRACT}" --pair-database "${RUN_ROOT}/lsh_pairs.sqlite" --shingle-manifest "${RUN_ROOT}/shingle_manifest.json" --runtime-receipt "${RUN_ROOT}/datatrove_runtime.json" --task-index "${TASK_INDEX}" --tasks "${VERIFY_TASKS}"
    ;;
  merge-verified)
    "${PYTHON}" "${DEDUP}" merge-verified --contract "${CONTRACT}" --pair-database "${RUN_ROOT}/lsh_pairs.sqlite" --tasks "${VERIFY_TASKS}" --output "${RUN_ROOT}/verified_manifest.json"
    ;;
  cluster)
    "${PYTHON}" "${DEDUP}" cluster --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --exact-manifest "${RUN_ROOT}/exact_manifest.json" --verified-manifest "${RUN_ROOT}/verified_manifest.json" --output "${RUN_ROOT}/removals.sqlite"
    ;;
  filter)
    "${PYTHON}" "${DEDUP}" filter-task --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --removal-database "${RUN_ROOT}/removals.sqlite" --task-index "${TASK_INDEX}"
    ;;
  merge-filtered)
    "${PYTHON}" "${DEDUP}" merge-filtered-release --contract "${CONTRACT}" --combined-manifest "${RUN_ROOT}/release-pre-dedup/manifests/combined_manifest.json" --removal-database "${RUN_ROOT}/removals.sqlite" --output "${RUN_ROOT}/release-deduplicated"
    ;;
  publish-dedup)
    : "${HF_TOKEN_FILE:?HF_TOKEN_FILE is required for publication}"
    test "$(stat -c %a "${HF_TOKEN_FILE}")" = "600" || { echo "HF token file must have mode 600" >&2; exit 2; }
    HF_TOKEN="$(<"${HF_TOKEN_FILE}")" "${PYTHON}" "${PUBLISH}" --release "${RUN_ROOT}/release-deduplicated" --kind deduplicated --repo-id "$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["private_repositories"]["deduplicated"])' "${CONTRACT}")" --staging "${RUN_ROOT}/publication-staging/deduplicated" --output "${RUN_ROOT}/publication-deduplicated.json" --execute
    ;;
  *)
    echo "unknown STAGE=${STAGE}" >&2
    exit 2
    ;;
esac
