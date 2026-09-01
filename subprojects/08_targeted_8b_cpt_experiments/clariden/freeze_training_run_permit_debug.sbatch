#!/usr/bin/env bash
#SBATCH --job-name=h2g-run-permit
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_SCALE:?set 8b or 1p5b}"
: "${H2G_PRODUCER_COMPATIBILITY:?set the exact accepted-producer authority}"
: "${H2G_PROFILE_PROMOTION:?set the exact profile-promotion receipt}"
: "${H2G_LR_SELECTION:?set the exact LR-selection receipt}"
: "${H2G_TRAINING_RUN_PERMIT_OUTPUT:?set a new immutable training-run-permit output path}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "training run permit requires one debug node" >&2; exit 2; }
for path in "$H2G_PRODUCER_COMPATIBILITY" "$H2G_PROFILE_PROMOTION" "$H2G_LR_SELECTION" "$H2G_TRAINING_RUN_PERMIT_OUTPUT"; do
  case "$path" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "run-permit inputs and output must be inside the stage receipts root: $path" >&2; exit 2 ;; esac
done
case "$H2G_SCALE" in 8b|1p5b) ;; *) echo "invalid scale" >&2; exit 2 ;; esac
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
out="$H2G_TRAINING_RUN_PERMIT_OUTPUT"
[[ ! -e "$out" ]] || { echo "immutable training run permit exists: $out" >&2; exit 2; }
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/build_training_run_permit.py" \
  --scale "$H2G_SCALE" --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" \
  --profile-promotion "$H2G_PROFILE_PROMOTION" \
  --lr-selection "$H2G_LR_SELECTION" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --output "$out"
