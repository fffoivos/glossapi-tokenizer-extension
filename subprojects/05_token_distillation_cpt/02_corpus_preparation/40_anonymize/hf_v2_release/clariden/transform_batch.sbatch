#!/usr/bin/env bash
#SBATCH --job-name=hf-v2-anon-batch
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=01:25:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
: "${HFV2_CODE_ROOT:?}"
: "${HFV2_RUN_ROOT:?}"
: "${HFV2_DATA_PYTHON:?}"
pipeline="$HFV2_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/hf_v2_release/scripts/pipeline.py"
stop_file="$HFV2_RUN_ROOT/control/transform_stop_requested"
mkdir -p "$(dirname "$stop_file")"
rm -f "$stop_file"
requeue_requested=0
request_requeue() { requeue_requested=1; : > "$stop_file"; }
request_stop() { : > "$stop_file"; }
trap request_requeue USR1
trap request_stop TERM INT
children=()
for lane in 0 1 2 3; do
  (
    uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
      "$HFV2_DATA_PYTHON" "$pipeline" transform-batch \
      --code-root "$HFV2_CODE_ROOT" --contract "$HFV2_RUN_ROOT/run_contract.json" \
      --lane "$lane" --lanes 4 --workers "${HFV2_WORKERS_PER_LANE:-64}" \
      --chunksize "${HFV2_CHUNKSIZE:-8}" --stop-file "$stop_file" || {
        : > "$stop_file"
        exit 1
      }
  ) &
  children+=("$!")
done
status=0
for child in "${children[@]}"; do
  wait "$child" || status=1
done
if ((requeue_requested)); then
  rm -f "$stop_file"
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
rm -f "$stop_file"
exit "$status"
