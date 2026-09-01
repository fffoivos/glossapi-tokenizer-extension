#!/usr/bin/env bash
#SBATCH --job-name=hf-v2-make-public
#SBATCH --account=a0140
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
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
script="$HFV2_CODE_ROOT/subprojects/05_token_distillation_cpt/02_corpus_preparation/40_anonymize/hf_v2_release/scripts/ensure_public_release.py"
uenv run pytorch/v2.6.0:v1 --view=default -- env -u PYTHONPATH -u PYTHONHOME \
  HF_HUB_DISABLE_XET=1 HF_TOKEN="$HF_TOKEN" "$HFV2_HF_PYTHON" "$script" \
  --repo-id fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2 \
  --expected-sha 987b8955fcd395c6219e39df9e64715457f69065 \
  --expected-manifest-sha256 1343b49211f84ab86553476824321d102fe31fb06fad36391e1134fa428f0114 \
  --receipt "$HFV2_RUN_ROOT/publication/public_access_receipt.json" \
  --execute
