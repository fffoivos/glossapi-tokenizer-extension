#!/usr/bin/env bash
#SBATCH --job-name=h2g-profile-finalize
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
  H2G_BENCHMARK_OUTPUT_ROOT H2G_REFERENCE_BENCHMARK_CONTRACT \
  H2G_REFERENCE_THROUGHPUT_LOG H2G_PROFILE_BENCHMARK_OUTPUT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "profile finalization requires one debug node" >&2; exit 2; }
case "$H2G_PROFILE_BENCHMARK_OUTPUT" in "$H2G_STAGE_ROOT"/receipts/*) ;; *) echo "profile-benchmark output must stay inside the stage receipts root" >&2; exit 2 ;; esac
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
read_contract() { /usr/bin/python3.11 - "$H2G_BENCHMARK_CONTRACT" "$1" <<'PY'
import json,sys
value=json.load(open(sys.argv[1]))
for key in sys.argv[2].split('.'):
    value=value[key]
print(value)
PY
}
scale=$(read_contract scale)
profile=$(read_contract profile_id)
nodes=$(read_contract nodes)
tp=$(read_contract tensor_parallel)
microbatch=$(read_contract microbatch)
out="$H2G_PROFILE_BENCHMARK_OUTPUT"
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/finalize_profile_benchmark.py" \
  --scale "$scale" --profile-id "$profile" --nodes "$nodes" --tensor-parallel "$tp" --microbatch "$microbatch" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" \
  --benchmark-contract "$H2G_BENCHMARK_CONTRACT" \
  --preflight "$H2G_BENCHMARK_OUTPUT_ROOT/preflight.json" \
  --throughput-log "$H2G_BENCHMARK_OUTPUT_ROOT/throughput/driver.out" \
  --reference-benchmark-contract "$H2G_REFERENCE_BENCHMARK_CONTRACT" \
  --reference-throughput-log "$H2G_REFERENCE_THROUGHPUT_LOG" \
  --uninterrupted-log "$H2G_BENCHMARK_OUTPUT_ROOT/restart/uninterrupted/driver.out" \
  --resumed-log "$H2G_BENCHMARK_OUTPUT_ROOT/restart/resumed/driver.out" \
  --phase2-uninterrupted-log "$H2G_BENCHMARK_OUTPUT_ROOT/restart/phase2_uninterrupted/driver.out" \
  --phase2-resumed-log "$H2G_BENCHMARK_OUTPUT_ROOT/restart/phase2_resumed/driver.out" \
  --output "$out"
