#!/usr/bin/env bash
#SBATCH --job-name=h2g-stage-b
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:25:00
#SBATCH --no-requeue

set -euo pipefail
: "${H2G_CODE_ROOT:?set immutable code root}"
: "${H2G_CODE_RECEIPT:?set immutable code receipt}"
: "${H2G_STAGE_ROOT:?set immutable stage root}"
: "${H2G_STAGE_B_STREAM:?set hplt, openarchives, or replay_selected}"
data_python=${H2G_DATA_PYTHON:?set verified bundle-bound data runtime Python}
data_runtime_root=${H2G_DATA_RUNTIME_ROOT:?set verified bundle-bound data runtime root}
case "$H2G_STAGE_B_STREAM" in
  hplt|openarchives)
    mode=assert_noop
    input="$H2G_STAGE_ROOT/data/${H2G_STAGE_B_STREAM}_prepared_clean.jsonl"
    input_receipt="$H2G_STAGE_ROOT/receipts/${H2G_STAGE_B_STREAM}_prepared_stream.json"
    ;;
  replay_selected)
    mode=apply
    input="$H2G_STAGE_ROOT/data/greekmmlu_scans/replay_selected/clean.jsonl"
    input_receipt="$H2G_STAGE_ROOT/receipts/greekmmlu_scan_replay_selected.json"
    ;;
  *) echo "invalid Stage-B stream: $H2G_STAGE_B_STREAM" >&2; exit 2 ;;
esac
[[ "${SLURM_JOB_PARTITION:-}" == debug && "${SLURM_NNODES:-0}" == 1 ]] || { echo "Stage-B preparation requires one debug node" >&2; exit 2; }
/usr/bin/python3.11 "$H2G_CODE_ROOT/subprojects/06_dataset_scheduling_experiments/production/verify_code_bundle.py" \
  --root "$H2G_CODE_ROOT" --receipt "$H2G_CODE_RECEIPT" --kind scientific
source "$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments/clariden/verify_data_runtime.inc.sh"
subproject="$H2G_CODE_ROOT/subprojects/08_targeted_8b_cpt_experiments"
uenv run pytorch/v2.9.1:v2 --view=default -- \
  "$data_python" "$subproject/scripts/anonymize_training_stream.py" \
    --stream "$H2G_STAGE_B_STREAM" --mode "$mode" \
    --input-jsonl "$input" --input-receipt "$input_receipt" \
    --masker-script "$H2G_CODE_ROOT/frozen_historical_dataset_tools/pii_masker.py" \
    --output-jsonl "$H2G_STAGE_ROOT/data/${H2G_STAGE_B_STREAM}_stage_b.jsonl" \
    --output-receipt "$H2G_STAGE_ROOT/receipts/${H2G_STAGE_B_STREAM}_stage_b.json"
