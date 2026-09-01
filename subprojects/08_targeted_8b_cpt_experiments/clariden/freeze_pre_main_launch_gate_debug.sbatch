#!/usr/bin/env bash
#SBATCH --job-name=h2g-launch-gate
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_PRODUCER_COMPATIBILITY \
  H2G_SCALE H2G_ARTIFACT_MANIFEST H2G_LAUNCH_GATE_OUTPUT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "$H2G_SCALE" == 8b || "$H2G_SCALE" == 1p5b ]] || { echo "H2G_SCALE must be 8b or 1p5b" >&2; exit 2; }
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "launch-gate freeze requires one debug node" >&2; exit 2; }
for path in "$H2G_ARTIFACT_MANIFEST" "$H2G_LAUNCH_GATE_OUTPUT"; do
  case "$path" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "launch-gate input and output must stay inside the stage receipts root: $path" >&2; exit 2 ;; esac
done
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
manifest="$H2G_ARTIFACT_MANIFEST"
output="$H2G_LAUNCH_GATE_OUTPUT"
[[ -s "$manifest" && ! -e "$output" ]] || { echo "final artifact manifest missing or launch gate exists" >&2; exit 2; }
export H2G_CODE_ROOT H2G_CODE_RECEIPT
/usr/bin/python3.11 "$subproject/scripts/freeze_hard_h_to_g_contract.py" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" \
  --artifact-manifest "$manifest" --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --mode launch --gate-stage pre_main --scale "$H2G_SCALE" --output "$output"
