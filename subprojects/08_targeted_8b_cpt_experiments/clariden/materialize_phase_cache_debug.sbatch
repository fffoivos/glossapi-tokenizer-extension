#!/usr/bin/env bash
#SBATCH --job-name=h2g-cache-materialize
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_CACHE_ROLE H2G_PHASE \
  H2G_SOURCE_CACHE_RECEIPT H2G_PRODUCER_COMPATIBILITY H2G_CACHE_DESTINATION_ROOT \
  H2G_CACHE_MATERIALIZATION_OUTPUT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "phase cache materialization requires one debug node" >&2; exit 2;
}
case "$H2G_CACHE_ROLE" in canonical|qualification-overlay) ;; *) echo "invalid cache role" >&2; exit 2 ;; esac
case "$H2G_CACHE_MATERIALIZATION_OUTPUT" in "$H2G_STAGE_ROOT"/receipts/*) ;; *)
  echo "cache materialization output must stay inside the stage receipts root" >&2; exit 2 ;;
esac
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

failed_root="$H2G_STAGE_ROOT/_failed/cache-materialize-${H2G_CACHE_ROLE}-phase-${H2G_PHASE}-${SLURM_JOB_ID}"
quarantine_failure() {
  mkdir -p "$failed_root"
  if [[ -e "$H2G_CACHE_DESTINATION_ROOT" ]]; then mv -- "$H2G_CACHE_DESTINATION_ROOT" "$failed_root/cache_root"; fi
  if [[ -e "$H2G_CACHE_MATERIALIZATION_OUTPUT" ]]; then mv -- "$H2G_CACHE_MATERIALIZATION_OUTPUT" "$failed_root/"; fi
}
trap quarantine_failure ERR
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
export H2G_CODE_ROOT H2G_CODE_RECEIPT
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/materialize_phase_cache.py" \
  --role "$H2G_CACHE_ROLE" --phase "$H2G_PHASE" \
  --source-receipt "$H2G_SOURCE_CACHE_RECEIPT" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --destination-root "$H2G_CACHE_DESTINATION_ROOT" \
  --output "$H2G_CACHE_MATERIALIZATION_OUTPUT"
trap - ERR
