#!/usr/bin/env bash
#SBATCH --job-name=h2g-train
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue

set -euo pipefail
if [[ -n "${H2G_SRUN_WRAPPER_DIR:-}" ]]; then
  export PATH="$H2G_SRUN_WRAPPER_DIR:$PATH"
fi
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_MODEL_SCALE:?set 8b or 1p5b}"
: "${H2G_PHASE:?set 1, 2 or 3}"
: "${H2G_START_UPDATE:?set exact start update}"
: "${H2G_EXIT_UPDATE:?set exact exit update}"
: "${H2G_LOAD_CHECKPOINT:?set init or resume checkpoint root}"
: "${H2G_PHASE_DATA_PATH_SPEC:?set exact frozen phase data-path spec}"
: "${H2G_PHASE_DATA_PATH:?set exact ordered weighted --data-path string}"
: "${H2G_PHASE_CACHE_RECEIPT:?set exact phase cache receipt}"
: "${H2G_PHASE_CACHE_ROOT:?set exact phase GPTDataset cache root}"
: "${H2G_PHASE_CACHE_TREE_SHA256:?set exact target phase cache tree SHA-256}"
: "${H2G_SEGMENT_OUTPUT_ROOT:?set new segment output root}"
: "${H2G_PEAK_LR:?set frozen peak LR}"
: "${H2G_FLOOR_LR:?set exact 0.1-times floor LR}"
: "${H2G_MICROBATCH_SIZE:?set promoted microbatch}"
: "${H2G_TENSOR_PARALLEL_SIZE:?set promoted tensor parallel}"
: "${H2G_VAL_DATA_DIR:?set frozen validation binary root}"
: "${H2G_ONLINE_VALIDATION_RECEIPT:?set frozen historical online-validation receipt}"
: "${H2G_EXTRA_VALID_SETS:?set frozen validation panel names}"
: "${H2G_NEW_GREEK_VALID_SETS:?set frozen Greek panel names}"
: "${H2G_MEGATRON_DIR:?set patched pinned Megatron root}"
: "${H2G_MEGATRON_RECEIPT:?set frozen patched Megatron receipt}"
: "${H2G_TRAINING_RUN_PERMIT:?set receipt binding promoted profile and selected LR}"
: "${H2G_AUTHORIZATION_GATE:?set the launch-ready gate for this segment stage}"
: "${H2G_STATIC_PREFLIGHT:?set the passed preallocation static-preflight receipt}"
[[ "${SLURM_JOB_PARTITION:-}" == normal ]] || { echo "training requires normal" >&2; exit 2; }
[[ "${SLURM_NNODES:-0}" =~ ^[0-9]+$ ]] || exit 2
[[ ! -e "$H2G_SEGMENT_OUTPUT_ROOT" ]] || { echo "segment output exists" >&2; exit 2; }

/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
mkdir -p "$(dirname "$H2G_SEGMENT_OUTPUT_ROOT")" "$H2G_STAGE_ROOT/receipts/train_preflight"
runtime_compat_dir="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/runtime_compat"
attempt_tag="${H2G_ATTEMPT_TAG:-$SLURM_JOB_ID}"
[[ "$attempt_tag" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid H2G_ATTEMPT_TAG" >&2; exit 2; }
runtime_compat_receipt="$H2G_STAGE_ROOT/receipts/train_preflight/dcp_metadata_compat_${attempt_tag}.json"
[[ -f "$runtime_compat_dir/sitecustomize.py" ]] || {
  echo "missing frozen DCP metadata compatibility shim" >&2
  exit 2
}
runtime_compat_command=(
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$runtime_compat_dir"
  python3 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/check_dcp_metadata_compat.py"
  --runtime-compat-dir "$runtime_compat_dir" --output "$runtime_compat_receipt"
)
if [[ -n "${UENV_MOUNT_LIST:-}" ]]; then
  # The canonical direct-salloc controller already runs in the manifest's
  # verified uenv. Re-entering it fails before model load; execute this
  # metadata-only compatibility probe in the mounted environment instead.
  "${runtime_compat_command[@]}"
else
  uenv run pytorch/v2.9.1:v2 --view=default -- "${runtime_compat_command[@]}"
fi
preflight="$H2G_STAGE_ROOT/receipts/train_preflight/${H2G_MODEL_SCALE}_p${H2G_PHASE}_${H2G_START_UPDATE}_${H2G_EXIT_UPDATE}_${attempt_tag}.json"
preflight_args=(
  --experiment "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/configs/hard_h_to_g_replication_v1.json"
  --scale "$H2G_MODEL_SCALE" --phase "$H2G_PHASE"
  --start-update "$H2G_START_UPDATE" --exit-update "$H2G_EXIT_UPDATE"
  --load-checkpoint "$H2G_LOAD_CHECKPOINT"
  --phase-data-path-spec "$H2G_PHASE_DATA_PATH_SPEC" --phase-data-path "$H2G_PHASE_DATA_PATH"
  --phase-cache-receipt "$H2G_PHASE_CACHE_RECEIPT"
  --phase-cache-root "$H2G_PHASE_CACHE_ROOT"
  --phase-cache-tree-sha256 "$H2G_PHASE_CACHE_TREE_SHA256"
  --megatron-root "$H2G_MEGATRON_DIR" --megatron-receipt "$H2G_MEGATRON_RECEIPT"
  --validation-root "$H2G_VAL_DATA_DIR" --validation-receipt "$H2G_ONLINE_VALIDATION_RECEIPT"
  --runtime-compat-dir "$runtime_compat_dir" --runtime-compat-receipt "$runtime_compat_receipt"
  --extra-valid-sets "$H2G_EXTRA_VALID_SETS" --new-greek-valid-sets "$H2G_NEW_GREEK_VALID_SETS"
  --nodes "$SLURM_NNODES" --tensor-parallel "$H2G_TENSOR_PARALLEL_SIZE"
  --microbatch "$H2G_MICROBATCH_SIZE" --output "$preflight"
  --training-run-permit "$H2G_TRAINING_RUN_PERMIT"
  --authorization-gate "$H2G_AUTHORIZATION_GATE"
  --static-preflight-receipt "$H2G_STATIC_PREFLIGHT"
  --peak-lr "$H2G_PEAK_LR" --floor-lr "$H2G_FLOOR_LR"
)
if (( H2G_START_UPDATE > 0 )); then
  : "${H2G_CHECKPOINT_PERMIT:?resumed segment requires a signed checkpoint permit}"
  : "${H2G_SOURCE_PHASE_CACHE_RECEIPT:?resumed segment requires the cache that produced its checkpoint}"
  preflight_args+=(
    --checkpoint-permit "$H2G_CHECKPOINT_PERMIT"
    --source-phase-cache-receipt "$H2G_SOURCE_PHASE_CACHE_RECEIPT"
  )
  if [[ "${H2G_CANONICAL_RESUME:-0}" == 1 ]]; then
    preflight_args+=(--canonical-resume)
  fi
else
  : "${H2G_INITIALIZATION_PERMIT:?initial segment requires the exact roundtrip receipt}"
  preflight_args+=(--initialization-permit "$H2G_INITIALIZATION_PERMIT")
fi
export H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_PHASE_DATA_PATH H2G_PHASE_CACHE_ROOT H2G_PHASE_CACHE_TREE_SHA256
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/scripts/preflight_train_segment.py" "${preflight_args[@]}"

if [[ -n "${H2G_RUNTIME_PHASE_CACHE_ROOT:-}" ]]; then
  : "${H2G_CACHE_OVERLAY_RECEIPT:?}"
  : "${H2G_PRODUCER_COMPATIBILITY:?}"
  /usr/bin/python3.11 - "$H2G_CODE_ROOT" "$H2G_CACHE_OVERLAY_RECEIPT" "$H2G_RUNTIME_PHASE_CACHE_ROOT" "$H2G_PRODUCER_COMPATIBILITY" <<'PY'
import sys
from pathlib import Path
code, receipt_path, overlay_root, authority_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(code / "subprojects/08_targeted_8b_cpt_experiments/scripts"))
from contract_utils import executing_code_bundle, read_json
from materialize_phase_cache import validate_overlay_receipt
from producer_bundle_compatibility import load_authority
_, accepted = load_authority(authority_path, executing_code_bundle())
validate_overlay_receipt(
    read_json(receipt_path), phase=3, overlay_root=overlay_root,
    accepted_code_bundles={(root, tree) for root, tree, *_ in accepted},
    require_pristine=False,
)
print("phase-3 runtime cache overlay verified", overlay_root)
PY
  export H2G_PHASE_CACHE_ROOT="$H2G_RUNTIME_PHASE_CACHE_ROOT"
fi

export ARM=td INIT_CKPT="$H2G_LOAD_CHECKPOINT" OUTPUT_DIR="$H2G_SEGMENT_OUTPUT_ROOT"
export RUNTIME_COMPAT_DIR="$runtime_compat_dir"
unset H2G_SAVE_INTERVAL_OVERRIDE H2G_EVAL_INTERVAL_OVERRIDE
export TRAIN_CONFIG_OVERRIDE="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/hard_h_to_g_training.env"
unset H2G_PHASE_START_UPDATE_OVERRIDE H2G_PRELAUNCH_PHASE2_RESTART_SMOKE
export SCRIPT_DIR_OVERRIDE="$H2G_CODE_ROOT/frozen_training_tools/bakeoff_training"
export ACCOUNT=a0140 PARTITION=normal NODES="$SLURM_NNODES" GPUS_PER_NODE=4 LAUNCH_MODE=torchrun
export TIME_LIMIT="${H2G_TIME_LIMIT:-12:00:00}"

request_graceful_stop() {
  mkdir -p "$H2G_SEGMENT_OUTPUT_ROOT/triggers"
  : > "$H2G_SEGMENT_OUTPUT_ROOT/triggers/save"
  : > "$H2G_SEGMENT_OUTPUT_ROOT/triggers/exit"
  echo "requested graceful save+exit for training child ${child_pid:-not_started}" >&2
}
trap request_graceful_stop USR1 TERM INT
bash "$SCRIPT_DIR_OVERRIDE/bakeoff_train.sbatch" &
child_pid=$!
set +e
wait "$child_pid"
rc=$?
set -e
if kill -0 "$child_pid" 2>/dev/null; then
  set +e
  wait "$child_pid"
  rc=$?
  set -e
fi
exit "$rc"
