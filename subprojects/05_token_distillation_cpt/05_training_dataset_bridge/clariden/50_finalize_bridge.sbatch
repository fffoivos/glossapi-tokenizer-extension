#!/usr/bin/env bash
#SBATCH --job-name=cpt5_finalize_bins
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${BRIDGE_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
bridge_require_paths
test -s "$INPUT_RECEIPT"
test -s "$HELDOUT_MANIFEST"
if [[ -s "$BRIDGE_MANIFEST" ]]; then
    bridge_python - "$BRIDGE_MANIFEST" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
assert value["schema_version"] == "full_cpt_training_bridge_manifest_v1"
assert value["status"] == "completed"
print(f"bridge already complete: {sys.argv[1]}")
PY
    exit 0
fi
bridge_python "$FINALIZE_SCRIPT" \
    --input-receipt "$INPUT_RECEIPT" \
    --heldout-manifest "$HELDOUT_MANIFEST" \
    --config "$BRIDGE_CONFIG" \
    --stage-root "$BRIDGE_STAGE_ROOT" \
    --output "$BRIDGE_MANIFEST"
