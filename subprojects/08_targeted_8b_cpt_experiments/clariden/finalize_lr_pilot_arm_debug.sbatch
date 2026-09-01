#!/usr/bin/env bash
#SBATCH --job-name=h2g-lr-arm-finalize
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_BENCHMARK_CONTRACT \
  H2G_BENCHMARK_OUTPUT_ROOT H2G_LR_PILOT_ARM_OUTPUT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "LR-arm finalization requires one debug node" >&2; exit 2; }
case "$H2G_LR_PILOT_ARM_OUTPUT" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "LR-pilot-arm output must stay inside the stage receipts root" >&2; exit 2 ;; esac
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
peak=$(/usr/bin/python3.11 - "$H2G_BENCHMARK_CONTRACT" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["peak_lr"])
PY
)
out="$H2G_LR_PILOT_ARM_OUTPUT"
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/finalize_lr_pilot_arm.py" \
  --benchmark-contract "$H2G_BENCHMARK_CONTRACT" \
  --preflight "$H2G_BENCHMARK_OUTPUT_ROOT/preflight.json" \
  --pilot-log "$H2G_BENCHMARK_OUTPUT_ROOT/pilot/driver.out" \
  --run-metadata "$H2G_BENCHMARK_OUTPUT_ROOT/pilot/run_metadata.json" \
  --checkpoint-root "$H2G_BENCHMARK_OUTPUT_ROOT/pilot/checkpoints" \
  --output "$out"
