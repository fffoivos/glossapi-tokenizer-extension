#!/usr/bin/env bash
#SBATCH --job-name=hf-v2-anon-publish
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:25:00
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -euo pipefail
: "${HFV2_CODE_ROOT:?}"
: "${HFV2_RUN_ROOT:?}"
: "${HFV2_HF_PYTHON:?}"
: "${HFV2_HF_TOKEN_FILE:?}"
[[ -f "$HFV2_HF_TOKEN_FILE" && ! -L "$HFV2_HF_TOKEN_FILE" ]] || {
  echo "missing regular HF token file" >&2
  exit 2
}
[[ "$(stat -c %a "$HFV2_HF_TOKEN_FILE")" == 600 ]] || {
  echo "HF token file must have mode 600" >&2
  exit 2
}
HF_TOKEN=$(<"$HFV2_HF_TOKEN_FILE")
[[ -n "$HF_TOKEN" ]] || { echo "HF token file is empty" >&2; exit 2; }
publisher="$HFV2_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/hf_v2_release/scripts/publish_release.py"
requeue_requested=0
child=''
request_requeue() {
  requeue_requested=1
  if [[ -n "$child" ]]; then kill -INT "$child" 2>/dev/null || true; fi
}
trap request_requeue USR1 TERM INT
uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME HF_TOKEN="$HF_TOKEN" \
  "$HFV2_HF_PYTHON" "$publisher" \
  --code-root "$HFV2_CODE_ROOT" \
  --contract "$HFV2_RUN_ROOT/run_contract.json" \
  --release-root "$HFV2_RUN_ROOT/release" \
  --manifest "$HFV2_RUN_ROOT/release/manifests/anonymization_manifest.json" \
  --state "$HFV2_RUN_ROOT/publication/state.json" \
  --receipt "$HFV2_RUN_ROOT/publication/receipt.json" --execute &
child=$!
set +e
wait "$child"
status=$?
set -e
if ((requeue_requested)); then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$status"
