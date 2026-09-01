#!/usr/bin/env bash
#SBATCH --job-name=h2g-gmmlu-query
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code-bundle receipt}"
: "${H2G_STAGE_ROOT:?set immutable experiment stage root}"
H2G_EVAL_PYTHON=${H2G_EVAL_PYTHON:-/iopsstor/scratch/cscs/fffoivos/python_envs/native_greek_eval_py312_aug2026/bin/python}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "GreekMMLU query freeze requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
source_root="$H2G_CODE_ROOT/frozen_greekmmlu_query_builder"
queries="$H2G_STAGE_ROOT/data/greekmmlu_public_queries.jsonl"
summary="$H2G_STAGE_ROOT/receipts/greekmmlu_public_queries.summary.json"
receipt="$H2G_STAGE_ROOT/receipts/greekmmlu_query_regeneration.json"
parity="$H2G_STAGE_ROOT/receipts/legacy_public_greekmmlu_loader_parity.json"
for output in "$queries" "$summary" "$receipt" "$parity"; do
  [[ ! -e "$output" ]] || { echo "immutable GreekMMLU query output exists: $output" >&2; exit 2; }
done
mkdir -p "$H2G_STAGE_ROOT/data" "$H2G_STAGE_ROOT/receipts"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$H2G_EVAL_PYTHON" -c 'import datasets,pyarrow; print({"datasets":datasets.__version__,"pyarrow":pyarrow.__version__})'
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$H2G_EVAL_PYTHON" "$subproject/scripts/build_frozen_greekmmlu_queries.py" \
    --registry "$source_root/eval/native_greek_benchmark_registry.json" \
    --eval-dir "$source_root/eval" \
    --contract "$subproject/configs/greekmmlu_query_regeneration_v1.json" \
    --output-jsonl "$queries" \
    --summary-json "$summary"
/usr/bin/python3.11 "$subproject/scripts/audit_frozen_greekmmlu_queries.py" \
  --contract "$subproject/configs/greekmmlu_query_regeneration_v1.json" \
  --queries "$queries" \
  --summary "$summary" \
  --source-root "$source_root" \
  --output "$receipt"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  env H2G_CODE_ROOT="$H2G_CODE_ROOT" H2G_CODE_RECEIPT="$H2G_CODE_RECEIPT" \
  "$H2G_EVAL_PYTHON" "$subproject/scripts/freeze_legacy_greekmmlu_loader_parity.py" \
    --query-receipt "$receipt" \
    --current-evaluator "$source_root/eval/run_native_greek_mcq_eval.py" \
    --current-registry "$source_root/eval/native_greek_benchmark_registry.json" \
    --legacy-evaluator "$H2G_CODE_ROOT/frozen_legacy_greekmmlu_eval/run_native_greek_mcq_eval.py" \
    --legacy-registry "$H2G_CODE_ROOT/frozen_legacy_greekmmlu_eval/native_greek_benchmark_registry.json" \
    --output "$parity"
