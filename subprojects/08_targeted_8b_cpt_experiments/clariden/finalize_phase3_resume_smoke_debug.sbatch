#!/usr/bin/env bash
#SBATCH --job-name=h2g-p3-smoke-final
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_MODEL_SCALE H2G_START_UPDATE H2G_SMOKE_OUTPUT_ROOT H2G_PHASE3_CACHE_RECEIPT H2G_FLOOR_LR H2G_DATA_PYTHON H2G_DATA_RUNTIME_ROOT H2G_DATA_RUNTIME_RECEIPT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "Phase-3 smoke finalization requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
/usr/bin/python3.11 "$subproject/scripts/verify_data_runtime.py" \
  --runtime-root "$H2G_DATA_RUNTIME_ROOT" --receipt "$H2G_DATA_RUNTIME_RECEIPT" \
  --requirements "$subproject/configs/data_runtime_requirements_v1.txt"
end_update=$((H2G_START_UPDATE + 1))
checkpoint="$H2G_SMOKE_OUTPUT_ROOT/checkpoints/iter_$(printf '%07d' "$end_update")"
audit="$H2G_STAGE_ROOT/receipts/phase3_smoke_checkpoint_audit_${H2G_MODEL_SCALE}_${H2G_START_UPDATE}.json"
receipt="$H2G_STAGE_ROOT/receipts/phase3_resume_smoke_${H2G_MODEL_SCALE}_${H2G_START_UPDATE}.json"
env PYTHONDONTWRITEBYTECODE=1 "$H2G_DATA_PYTHON" "$subproject/scripts/audit_training_checkpoint.py" \
  --scale "$H2G_MODEL_SCALE" --source-phase 3 --update "$end_update" \
  --checkpoint-root "$checkpoint" --source-phase-cache-receipt "$H2G_PHASE3_CACHE_RECEIPT" \
  --segment-preflight "$H2G_SMOKE_OUTPUT_ROOT/preflight.json" --training-log "$H2G_SMOKE_OUTPUT_ROOT/driver.out" \
  --output "$audit"
/usr/bin/python3.11 "$subproject/scripts/finalize_phase3_resume_smoke.py" \
  --scale "$H2G_MODEL_SCALE" --start-update "$H2G_START_UPDATE" --floor-lr "$H2G_FLOOR_LR" \
  --preflight "$H2G_SMOKE_OUTPUT_ROOT/preflight.json" --checkpoint-audit "$audit" \
  --training-log "$H2G_SMOKE_OUTPUT_ROOT/driver.out" --output "$receipt"
