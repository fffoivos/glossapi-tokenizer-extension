#!/usr/bin/env bash
#SBATCH --job-name=h2g-ext-gate
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_GATE_STAGE \
  H2G_PRODUCER_COMPATIBILITY; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ -z "${H2G_SCALE:-}" ]] || {
  echo "extension gates are joint and forbid H2G_SCALE" >&2; exit 2;
}
case "$H2G_GATE_STAGE" in
  pre_extension|pre_second_extension|pre_finalization) ;;
  *) echo "invalid extension gate stage" >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "extension launch-gate freeze requires one debug node" >&2; exit 2;
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
manifest="$H2G_STAGE_ROOT/receipts/artifact_manifest_${H2G_GATE_STAGE}.json"
output="$H2G_STAGE_ROOT/receipts/launch_gate_${H2G_GATE_STAGE}.json"
[[ -s "$manifest" && ! -e "$output" ]] || {
  echo "final extension manifest missing or launch gate exists" >&2; exit 2;
}
export H2G_CODE_ROOT H2G_CODE_RECEIPT PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3.11 "$subproject/scripts/freeze_hard_h_to_g_contract.py" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" \
  --artifact-manifest "$manifest" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --mode launch --gate-stage "$H2G_GATE_STAGE" --output "$output"
