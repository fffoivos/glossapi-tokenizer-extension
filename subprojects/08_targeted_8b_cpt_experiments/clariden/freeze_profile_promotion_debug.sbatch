#!/usr/bin/env bash
#SBATCH --job-name=h2g-profile-promote
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
: "${H2G_PROFILE_CANDIDATE_RECEIPTS:?set colon-separated benchmark receipts}"
: "${H2G_PROFILE_PROMOTION_OUTPUT:?set a new immutable profile-promotion output path}"
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "profile promotion requires one debug node" >&2; exit 2; }
case "$H2G_PROFILE_PROMOTION_OUTPUT" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "profile-promotion output must be inside the stage receipts root" >&2; exit 2 ;; esac
case "$H2G_SCALE" in 8b|1p5b) ;; *) echo "invalid scale" >&2; exit 2 ;; esac
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
out="$H2G_PROFILE_PROMOTION_OUTPUT"
[[ ! -e "$out" ]] || { echo "immutable profile promotion exists: $out" >&2; exit 2; }
IFS=: read -r -a paths <<< "$H2G_PROFILE_CANDIDATE_RECEIPTS"
candidate_args=()
for path in "${paths[@]}"; do [[ -s "$path" ]] || { echo "candidate receipt missing: $path" >&2; exit 2; }; candidate_args+=(--candidate "$path"); done
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/freeze_profile_promotion.py" \
  --scale "$H2G_SCALE" --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" "${candidate_args[@]}" --output "$out"
