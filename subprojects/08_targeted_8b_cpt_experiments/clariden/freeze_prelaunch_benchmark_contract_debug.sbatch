#!/usr/bin/env bash
#SBATCH --job-name=h2g-benchmark-contract
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
for name in H2G_CODE_ROOT H2G_CODE_RECEIPT H2G_STAGE_ROOT H2G_BENCHMARK_KIND H2G_SCALE H2G_PROFILE_ID H2G_NODES H2G_TENSOR_PARALLEL H2G_MICROBATCH H2G_PEAK_LR H2G_INITIALIZATION_PERMIT H2G_PRODUCER_COMPATIBILITY H2G_PHASE1_CACHE_RECEIPT H2G_PHASE1_CACHE_OVERLAY_RECEIPT H2G_PHASE2_CACHE_RECEIPT H2G_PHASE2_CACHE_OVERLAY_RECEIPT H2G_BENCHMARK_OUTPUT_ROOT H2G_BENCHMARK_CONTRACT; do
  [[ -n "${!name:-}" ]] || { echo "$name is required" >&2; exit 2; }
done
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "benchmark contract freeze requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
extra=()
if [[ "$H2G_BENCHMARK_KIND" == lr ]]; then : "${H2G_PROFILE_PROMOTION:?LR benchmark requires profile promotion}"; extra+=(--profile-promotion "$H2G_PROFILE_PROMOTION"); fi
env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3.11 "$subproject/scripts/freeze_prelaunch_benchmark_contract.py" \
  --kind "$H2G_BENCHMARK_KIND" --scale "$H2G_SCALE" --profile-id "$H2G_PROFILE_ID" \
  --nodes "$H2G_NODES" --tensor-parallel "$H2G_TENSOR_PARALLEL" --microbatch "$H2G_MICROBATCH" \
  --peak-lr "$H2G_PEAK_LR" --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" --initialization-permit "$H2G_INITIALIZATION_PERMIT" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --phase1-cache-receipt "$H2G_PHASE1_CACHE_RECEIPT" \
  --phase1-cache-overlay-receipt "$H2G_PHASE1_CACHE_OVERLAY_RECEIPT" "${extra[@]}" \
  --phase2-cache-receipt "$H2G_PHASE2_CACHE_RECEIPT" \
  --phase2-cache-overlay-receipt "$H2G_PHASE2_CACHE_OVERLAY_RECEIPT" \
  --output-root "$H2G_BENCHMARK_OUTPUT_ROOT" --output "$H2G_BENCHMARK_CONTRACT"
