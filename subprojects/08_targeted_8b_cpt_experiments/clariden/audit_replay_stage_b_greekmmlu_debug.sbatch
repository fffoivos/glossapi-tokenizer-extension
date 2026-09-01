#!/usr/bin/env bash
#SBATCH --job-name=h2g-replay-gmmlu-post
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}

[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || {
  echo "post-Stage-B GreekMMLU audit requires one debug node" >&2
  exit 2
}
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"

subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" "$subproject/scripts/run_fresh_greekmmlu_stream_scan.py" \
    --stream-name replay_selected_post \
    --corpus-jsonl "$H2G_STAGE_ROOT/data/replay_selected_stage_b.jsonl" \
    --queries-jsonl "$H2G_STAGE_ROOT/data/greekmmlu_public_queries.jsonl" \
    --query-receipt "$H2G_STAGE_ROOT/receipts/greekmmlu_query_regeneration.json" \
    --scanner-script "$H2G_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/30_decontaminate/scripts/decontaminate.py" \
    --output-root "$H2G_STAGE_ROOT/data/greekmmlu_scans/replay_selected_post" \
    --workers "${SLURM_CPUS_PER_TASK:-64}" \
    --output-receipt "$H2G_STAGE_ROOT/receipts/greekmmlu_scan_replay_selected_post.json" \
    --audit-only
