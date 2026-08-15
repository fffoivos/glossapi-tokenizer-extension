#!/usr/bin/env bash
#SBATCH --job-name=h2g-eval-contracts
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code-bundle receipt}"
: "${H2G_STAGE_ROOT:?set immutable experiment stage root}"
: "${H2G_PRODUCER_COMPATIBILITY:?set exact current producer compatibility receipt}"
: "${H2G_GREEKMMLU_QUERY_RECEIPT:=$H2G_STAGE_ROOT/receipts/greekmmlu_query_regeneration.json}"
: "${H2G_LEGACY_LOADER_PARITY_RECEIPT:=$H2G_STAGE_ROOT/receipts/legacy_public_greekmmlu_loader_parity.json}"

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "evaluation-contract freeze requires one debug node" >&2
  exit 2
}
freeze_code="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/freeze_code_bundle.py"
verify_code="$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py"
/usr/bin/python3.11 "$verify_code" --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
receipts="$H2G_STAGE_ROOT/receipts"
mkdir -p "$receipts"
names=(
  evaluation_code_bundle.json
  legacy_public_evaluator_contract.json
  statistical_decision_contract.json
  hard_h_to_g_planning_contract_8b.json
  hard_h_to_g_planning_contract_1p5b.json
)
for name in "${names[@]}"; do
  output="$receipts/$name"
  [[ ! -e "$output" ]] || { echo "immutable evaluation receipt exists: $output" >&2; exit 2; }
done
partial="$receipts/.evaluation_contracts.${SLURM_JOB_ID}.partial"
failed="$receipts/_failed_evaluation_contracts.${SLURM_JOB_ID}"
[[ ! -e "$partial" && ! -e "$failed" ]] || { echo "evaluation temporary path exists" >&2; exit 2; }
mkdir "$partial"
preserve_failed_group() {
  status=$?
  if (( status != 0 )) && [[ -d "$partial" ]]; then mv "$partial" "$failed"; fi
  exit "$status"
}
trap preserve_failed_group EXIT

/usr/bin/python3.11 "$freeze_code" \
  --root "$H2G_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation" \
  --kind scientific \
  --output "$partial/evaluation_code_bundle.json"
/usr/bin/python3.11 "$verify_code" \
  --root "$H2G_CODE_ROOT/subprojects/09_full_8b_cpt_results_analysis/evaluation" \
  --receipt "$partial/evaluation_code_bundle.json" \
  --kind scientific
/usr/bin/python3.11 "$subproject/scripts/freeze_legacy_public_evaluator.py" \
  --contract "$subproject/configs/legacy_public_greekmmlu_v1.json" \
  --source-root "$H2G_CODE_ROOT/frozen_legacy_greekmmlu_eval" \
  --query-receipt "$H2G_GREEKMMLU_QUERY_RECEIPT" \
  --loader-parity-receipt "$H2G_LEGACY_LOADER_PARITY_RECEIPT" \
  --snapshot-adapter "$subproject/scripts/run_legacy_greekmmlu_snapshot_eval.py" \
  --producer-compatibility "$H2G_PRODUCER_COMPATIBILITY" \
  --output "$partial/legacy_public_evaluator_contract.json"
/usr/bin/python3.11 "$subproject/scripts/freeze_statistical_decision_contract.py" \
  --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
  --output "$partial/statistical_decision_contract.json"
for scale in 8b 1p5b; do
  /usr/bin/python3.11 "$subproject/scripts/freeze_hard_h_to_g_contract.py" \
    --experiment "$subproject/configs/hard_h_to_g_replication_v1.json" \
    --allocation "$subproject/configs/hard_h_to_g_allocation_v1.json" \
    --mode planning --scale "$scale" \
    --output "$partial/hard_h_to_g_planning_contract_${scale}.json"
done
for name in "${names[@]}"; do mv "$partial/$name" "$receipts/$name"; done
rmdir "$partial"
trap - EXIT
