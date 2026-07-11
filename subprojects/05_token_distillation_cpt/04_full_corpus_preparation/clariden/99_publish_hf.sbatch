#!/usr/bin/env bash
#SBATCH --job-name=cpt4_publish
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${PHASE04_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
source "$HERE/common.sh"
phase04_require_compute_partition "explicit gated Hugging Face publication"
phase04_require_cpu_request
phase04_require_uenv_python
phase04_require_clean_git "$REPO_ROOT"
phase04_require_expected_commit "$REPO_ROOT"
phase04_require_file "$PUBLISH_RELEASE_SCRIPT"
[[ -n "${HF_TOKEN:-}" ]] || { echo "ERROR: pass HF_TOKEN only in the submission environment" >&2; exit 103; }
[[ "${CONFIRM_PUBLISH:-}" == "$HF_REPO_ID" ]] || {
    echo "ERROR: set CONFIRM_PUBLISH exactly to $HF_REPO_ID for this standalone stage." >&2
    exit 104
}
phase04_init_pipeline_run
phase04_stage_begin "90-publish"
[[ "$PHASE04_STAGE_ALREADY_DONE" == "0" ]] || exit 0
phase04_stage_require_upstream "80-materialize-validate"

RELEASE_DIR="$PIPELINE_RUN_ROOT/stages/80-materialize-validate"
phase04_stage_add_input release_manifest "$RELEASE_DIR/release_manifest.json"
phase04_stage_add_input release_validation "$RELEASE_DIR/validation.json"
# Never print the command environment or token.  Publication is manual-gated
# and new-empty only; widening either policy requires a reviewed code change.
if [[ ! -e "$PHASE04_STAGE_DIR/publication_receipt.json" ]]; then
    uenv run "$PHASE04_UENV" --view=default -- \
        "$RUNTIME_VENV/bin/python" "$PUBLISH_RELEASE_SCRIPT" \
        --release "$RELEASE_DIR/release" \
        --release-manifest "$RELEASE_DIR/release_manifest.json" \
        --validation-receipt "$RELEASE_DIR/validation.json" \
        --repo-id "$HF_REPO_ID" \
        --gate-mode manual \
        --remote-mode new-empty \
        --output "$PHASE04_STAGE_DIR/publication_receipt.json" \
        --execute
fi

uenv run "$PHASE04_UENV" --view=default -- \
    "$RUNTIME_VENV/bin/python" - "$PHASE04_STAGE_DIR/publication_receipt.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = json.loads(path.read_text(encoding="utf-8"))
if value.get("schema_version") != "full_cpt_publication_receipt_v1":
    raise SystemExit(f"ERROR: unsupported publication receipt schema: {path}")
if value.get("status") != "passed" or not value.get("commit_sha"):
    raise SystemExit(f"ERROR: publication receipt is not a completed remote publication: {path}")
if value.get("gate_mode") != "manual" or value.get("remote_mode") != "new-empty":
    raise SystemExit(f"ERROR: publication policy drift recorded in {path}")
PY

phase04_stage_finish "$PHASE04_STAGE_DIR/publication_receipt.json"
