#!/usr/bin/env bash
#SBATCH --job-name=h2g-owner-auth
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_AUTHORIZATION_STAGE \
  H2G_OWNER_CONFIRMATION H2G_OWNER_CONFIRMED_AT H2G_PREAUTHORIZATION_MANIFEST \
  H2G_PRODUCER_COMPATIBILITY H2G_OWNER_AUTHORIZATION_OUTPUT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "owner authorization freeze requires one debug node" >&2; exit 2; }
for path in "$H2G_PREAUTHORIZATION_MANIFEST" "$H2G_OWNER_AUTHORIZATION_OUTPUT"; do
  case "$path" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "owner-authorization input and output must stay inside the stage receipts root: $path" >&2; exit 2 ;; esac
done
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
case "$H2G_AUTHORIZATION_STAGE" in
  pre_main)
    [[ "${H2G_SCALE:-}" == 8b || "${H2G_SCALE:-}" == 1p5b ]] || { echo "pre_main requires H2G_SCALE=8b or 1p5b" >&2; exit 2; }
    output="$H2G_OWNER_AUTHORIZATION_OUTPUT"
    scale_args=(--scale "$H2G_SCALE")
    ;;
  pre_extension)
    [[ -z "${H2G_SCALE:-}" ]] || { echo "pre_extension is a joint gate and forbids H2G_SCALE" >&2; exit 2; }
    output="$H2G_OWNER_AUTHORIZATION_OUTPUT"
    scale_args=()
    ;;
  *) echo "invalid authorization stage" >&2; exit 2 ;;
esac
export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 "$subproject/scripts/freeze_owner_authorization.py" \
  --authorization-stage "$H2G_AUTHORIZATION_STAGE" --owner fffoivos \
  "${scale_args[@]}" \
  --confirmation-text "$H2G_OWNER_CONFIRMATION" --confirmed-at "$H2G_OWNER_CONFIRMED_AT" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" \
  --preauthorization-manifest "$H2G_PREAUTHORIZATION_MANIFEST" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" --output "$output"
