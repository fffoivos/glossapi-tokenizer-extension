#!/usr/bin/env bash
#SBATCH --job-name=h2g-p3-resume-smoke
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue

set -euo pipefail
if [[ -n "${H2G_SRUN_WRAPPER_DIR:-}" ]]; then
  export PATH="$H2G_SRUN_WRAPPER_DIR:$PATH"
fi
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_MODEL_SCALE H2G_START_UPDATE H2G_LOAD_CHECKPOINT H2G_CHECKPOINT_PERMIT H2G_SOURCE_PHASE_CACHE_RECEIPT H2G_PHASE_DATA_PATH_SPEC H2G_PHASE_DATA_PATH H2G_PHASE_CACHE_RECEIPT H2G_PHASE_CACHE_ROOT H2G_PHASE_CACHE_TREE_SHA256 H2G_SMOKE_OUTPUT_ROOT H2G_PEAK_LR H2G_FLOOR_LR H2G_MICROBATCH_SIZE H2G_TENSOR_PARALLEL_SIZE H2G_VAL_DATA_DIR H2G_ONLINE_VALIDATION_RECEIPT H2G_EXTRA_VALID_SETS H2G_NEW_GREEK_VALID_SETS H2G_MEGATRON_DIR H2G_MEGATRON_RECEIPT H2G_TRAINING_RUN_PERMIT H2G_AUTHORIZATION_GATE H2G_STATIC_PREFLIGHT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == normal ]] || { echo "Phase-3 resume smoke requires normal" >&2; exit 2; }
[[ "$H2G_START_UPDATE" == 3218 || "$H2G_START_UPDATE" == 3456 ]] || { echo "invalid Phase-3 smoke start" >&2; exit 2; }
[[ ! -e "$H2G_SMOKE_OUTPUT_ROOT" ]] || { echo "immutable smoke output exists" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
end_update=$((H2G_START_UPDATE + 1))
runtime_compat_dir="$subproject/runtime_compat"
smoke_attempt_tag="${H2G_SMOKE_ATTEMPT_TAG:-$SLURM_JOB_ID}"
[[ "$smoke_attempt_tag" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "invalid H2G_SMOKE_ATTEMPT_TAG" >&2; exit 2; }
runtime_compat_receipt="$H2G_STAGE_ROOT/receipts/phase3_smoke_dcp_metadata_compat_${H2G_MODEL_SCALE}_${H2G_START_UPDATE}_${smoke_attempt_tag}.json"
runtime_compat_command=(
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$runtime_compat_dir"
  python3 "$subproject/scripts/check_dcp_metadata_compat.py"
  --runtime-compat-dir "$runtime_compat_dir" --output "$runtime_compat_receipt"
)
if [[ -n "${UENV_MOUNT_LIST:-}" ]]; then
  "${runtime_compat_command[@]}"
else
  uenv run pytorch/v2.9.1:v2 --view=default -- "${runtime_compat_command[@]}"
fi
preflight="$H2G_STAGE_ROOT/receipts/phase3_smoke_preflight_${H2G_MODEL_SCALE}_${H2G_START_UPDATE}_${smoke_attempt_tag}.json"
/usr/bin/python3.11 "$subproject/scripts/preflight_train_segment.py" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --scale "$H2G_MODEL_SCALE" --phase 3 --start-update "$H2G_START_UPDATE" --exit-update "$end_update" \
  --load-checkpoint "$H2G_LOAD_CHECKPOINT" --checkpoint-permit "$H2G_CHECKPOINT_PERMIT" \
  --source-phase-cache-receipt "$H2G_SOURCE_PHASE_CACHE_RECEIPT" \
  --phase-data-path-spec "$H2G_PHASE_DATA_PATH_SPEC" --phase-data-path "$H2G_PHASE_DATA_PATH" \
  --phase-cache-receipt "$H2G_PHASE_CACHE_RECEIPT" --phase-cache-root "$H2G_PHASE_CACHE_ROOT" \
  --phase-cache-tree-sha256 "$H2G_PHASE_CACHE_TREE_SHA256" \
  --megatron-root "$H2G_MEGATRON_DIR" --megatron-receipt "$H2G_MEGATRON_RECEIPT" \
  --validation-root "$H2G_VAL_DATA_DIR" --validation-receipt "$H2G_ONLINE_VALIDATION_RECEIPT" \
  --runtime-compat-dir "$runtime_compat_dir" --runtime-compat-receipt "$runtime_compat_receipt" \
  --extra-valid-sets "$H2G_EXTRA_VALID_SETS" --new-greek-valid-sets "$H2G_NEW_GREEK_VALID_SETS" \
  --training-run-permit "$H2G_TRAINING_RUN_PERMIT" --peak-lr "$H2G_PEAK_LR" --floor-lr "$H2G_FLOOR_LR" \
  --authorization-gate "$H2G_AUTHORIZATION_GATE" \
  --static-preflight-receipt "$H2G_STATIC_PREFLIGHT" \
  --nodes "$SLURM_NNODES" --tensor-parallel "$H2G_TENSOR_PARALLEL_SIZE" --microbatch "$H2G_MICROBATCH_SIZE" \
  --one-update-resume-smoke --output "$preflight"

# The immutable phase-cache receipt freezes the scientific training indices.
# Historical-validation indices are derived into a separately receipted,
# pristine overlay so the runtime never writes into that frozen cache.
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
    read_json(receipt_path),
    phase=3,
    overlay_root=overlay_root,
    accepted_code_bundles={(root, tree) for root, tree, *_ in accepted},
    require_pristine=False,
)
print("pristine phase-3 cache overlay verified", overlay_root)
PY
  export H2G_PHASE_CACHE_ROOT="$H2G_RUNTIME_PHASE_CACHE_ROOT"
fi

export H2G_PHASE=3 H2G_EXIT_UPDATE="$end_update"
if [[ -n "${H2G_RUNTIME_COMPAT_DIR:-}" ]]; then
  export RUNTIME_COMPAT_DIR="$H2G_RUNTIME_COMPAT_DIR"
  export PYTHONPATH="$H2G_RUNTIME_COMPAT_DIR${PYTHONPATH:+:$PYTHONPATH}"
fi
export TRAIN_CONFIG_OVERRIDE="$subproject/clariden/hard_h_to_g_training.env"
export SCRIPT_DIR_OVERRIDE="$H2G_CODE_ROOT/frozen_training_tools/bakeoff_training"
export ARM=td INIT_CKPT="$H2G_LOAD_CHECKPOINT" OUTPUT_DIR="$H2G_SMOKE_OUTPUT_ROOT"
export ACCOUNT=a0140 PARTITION=normal NODES="$SLURM_NNODES" GPUS_PER_NODE=4 LAUNCH_MODE=torchrun
export H2G_SAVE_INTERVAL_OVERRIDE=1 H2G_EVAL_INTERVAL_OVERRIDE=1000000 FULL_CPT_LAUNCH_VERIFY=0
export TIME_LIMIT="${H2G_TIME_LIMIT:-01:30:00}"
mkdir -p "$(dirname "$H2G_SMOKE_OUTPUT_ROOT")"
started_epoch=$(date +%s)
bash "$SCRIPT_DIR_OVERRIDE/bakeoff_train.sbatch" > "$H2G_SMOKE_OUTPUT_ROOT.driver.partial" 2>&1
mv "$H2G_SMOKE_OUTPUT_ROOT.driver.partial" "$H2G_SMOKE_OUTPUT_ROOT/driver.out"
completed_epoch=$(date +%s)
printf '%s\n' "$((completed_epoch - started_epoch))" > "$H2G_SMOKE_OUTPUT_ROOT/wall_seconds.txt"
cp "$preflight" "$H2G_SMOKE_OUTPUT_ROOT/preflight.json"
chmod -R a-w "$H2G_SMOKE_OUTPUT_ROOT"
