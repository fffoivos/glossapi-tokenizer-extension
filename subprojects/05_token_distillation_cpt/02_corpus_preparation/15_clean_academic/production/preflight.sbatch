#!/usr/bin/env bash
#SBATCH --job-name=bibclean-v2-preflight
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
: "${RUN_BOOTSTRAP:?RUN_BOOTSTRAP is required}"
: "${RELEASE_ROOT:?RELEASE_ROOT is required}"
: "${PUBLICATION_RECEIPT:?PUBLICATION_RECEIPT is required}"
: "${PREFLIGHT_OUTPUT:?PREFLIGHT_OUTPUT is required}"
export PYTHONPATH="$RUN_BOOTSTRAP/train${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${HF_TOKEN_FILE:-}" ]; then
  test "$(stat -c %a "$HF_TOKEN_FILE")" = "600" || {
    echo "HF token file must have mode 600" >&2
    exit 2
  }
  export HF_TOKEN
  HF_TOKEN="$(<"$HF_TOKEN_FILE")"
fi
exec "$HOME/.venvs/hfupload/bin/python" -m \
  subprojects.05_token_distillation_cpt.02_corpus_preparation.15_clean_academic.production.preflight \
  --release "$RELEASE_ROOT" \
  --publication-receipt "$PUBLICATION_RECEIPT" \
  --output "$PREFLIGHT_OUTPUT"
