#!/usr/bin/env bash
#SBATCH --job-name=t8b-validate
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
: "${TARGET8_CODE_ROOT:?set immutable code root}"
: "${TARGET8_CODE_RECEIPT:?set immutable code receipt}"
: "${TARGET8_RELEASE_ROOT:?set anonymized release root}"
: "${TARGET8_STAGE_ROOT:?set targeted preparation root}"
DATA_RUNTIME_ROOT=${TARGET8_DATA_RUNTIME_ROOT:?set immutable verified data runtime root}
DATA_PYTHON=${TARGET8_DATA_PYTHON:-$DATA_RUNTIME_ROOT/bin/python}
DATA_RUNTIME_RECEIPT=${TARGET8_DATA_RUNTIME_RECEIPT:-$DATA_RUNTIME_ROOT/runtime_receipt.json}
[[ "${SLURM_JOB_PARTITION:-}" == debug ]] || { echo "validation must run on debug" >&2; exit 2; }
[[ -s "$DATA_RUNTIME_RECEIPT" ]] || { echo "data runtime receipt missing" >&2; exit 2; }

verify="$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify" --root "$TARGET8_CODE_ROOT" --receipt "$TARGET8_CODE_RECEIPT" --kind scientific
export H2G_CODE_ROOT="$TARGET8_CODE_ROOT" H2G_CODE_RECEIPT="$TARGET8_CODE_RECEIPT"

subproject="$TARGET8_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
bundle_tag=$(/usr/bin/python3.11 - "$TARGET8_CODE_RECEIPT" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
tree=str(value.get("tree_sha256", ""))
if len(tree) != 64:
    raise SystemExit("scientific bundle receipt lacks tree hash")
print(tree)
PY
)
validation_root="$TARGET8_STAGE_ROOT/validation_runs/$bundle_tag"
[[ ! -e "$validation_root" ]] || { echo "immutable validation run exists: $validation_root" >&2; exit 2; }
mkdir -p "$validation_root"
cleanup_failed_validation() { rm -rf -- "$validation_root"; }
trap cleanup_failed_validation ERR
export PYTHONDONTWRITEBYTECODE=1
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" "$subproject/scripts/verify_data_runtime.py" \
    --runtime-root "$DATA_RUNTIME_ROOT" \
    --receipt "$DATA_RUNTIME_RECEIPT" \
    --requirements "$subproject/configs/data_runtime_requirements_v1.txt" \
    --code-root "$TARGET8_CODE_ROOT" \
    --code-receipt "$TARGET8_CODE_RECEIPT" \
  > "$validation_root/data_runtime_verification.json"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" -m pytest -q -p no:cacheprovider "$subproject/tests" \
  2>&1 | tee "$validation_root/tests.log"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" \
  "$TARGET8_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/tests/test_source_local_packing.py" \
  2>&1 | tee "$validation_root/source_local_packing_tests.log"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" "$subproject/scripts/validate_static_contracts.py" \
  > "$validation_root/static_contract_validation.json"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" "$subproject/scripts/inspect_hf_release.py" \
    --release-root "$TARGET8_RELEASE_ROOT" \
    --output "$validation_root/hf_release_inspection.json"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" "$subproject/scripts/freeze_experiment_contract.py" \
    --experiment-a-config "$subproject/configs/experiment_a_recipe.json" \
    --experiment-b-config "$subproject/configs/experiment_b_recipe.json" \
    --allocation-config "$subproject/configs/allocation_plan.json" \
    --mode planning \
    --output "$validation_root/planning_contracts.json"
for scale in 8b 1p5b; do
  uenv run pytorch/v2.9.1:v2 --view=default -- \
    "$DATA_PYTHON" "$subproject/scripts/freeze_hard_h_to_g_contract.py" \
      --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
      --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" \
      --mode planning --gate-stage pre_main --scale "$scale" \
      --output "$validation_root/hard_h_to_g_planning_contract_${scale}.json"
done
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$DATA_PYTHON" "$subproject/scripts/freeze_statistical_decision_contract.py" \
    --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
    --output "$validation_root/hard_h_to_g_statistical_contract.json"
trap - ERR
