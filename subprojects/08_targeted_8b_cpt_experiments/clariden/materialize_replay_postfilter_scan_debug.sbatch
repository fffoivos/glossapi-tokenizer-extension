#!/usr/bin/env bash
#SBATCH --job-name=h2g-replay-postscan
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:20:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "post-filter replay materialization requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
input="$H2G_STAGE_ROOT/data/replay_selected_stage_b.jsonl"
input_receipt="$H2G_STAGE_ROOT/receipts/replay_selected_stage_b.json"
greekmmlu_receipt="$H2G_STAGE_ROOT/receipts/greekmmlu_scan_replay_selected_post.json"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" "$subproject/scripts/materialize_replay_scan_corpus.py" \
    --input-jsonl "$input" \
    --input-receipt "$input_receipt" \
    --zero-greekmmlu-receipt "$greekmmlu_receipt" \
    --output-root "$H2G_STAGE_ROOT/data/replay_native_postfilter_scan_corpus" \
    --shards 32 \
    --output-receipt "$H2G_STAGE_ROOT/receipts/replay_native_postfilter_scan_corpus.json"
