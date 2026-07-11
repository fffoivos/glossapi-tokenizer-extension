#!/usr/bin/env bash
#SBATCH --job-name=cpt5_freeze_inputs
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail

HERE=${BRIDGE_CLARIDEN_DIR:?submit through clariden/submit.sh}
source "$HERE/paths.env"
bridge_require_paths
mkdir -p "$BRIDGE_STAGE_ROOT" "$BRIDGE_RUN_ROOT"

if [[ -s "$INPUT_RECEIPT" ]]; then
    bridge_python - "$INPUT_RECEIPT" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
assert value["schema_version"] == "full_cpt_training_bridge_input_receipt_v1"
assert value["status"] == "completed"
print(f"input receipt already complete: {sys.argv[1]}")
PY
    exit 0
fi

export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bridge_python "$FREEZE_SCRIPT" \
    --config "$BRIDGE_CONFIG" \
    --phase04-stage "$PHASE04_STAGE" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --scratch-root "$SCRATCH_ROOT" \
    --megatron-dir "$MEGATRON_DIR" \
    --replay-acquisition-receipt "$REPLAY_ACQUISITION_RECEIPT" \
    --old-greek-build-receipt "$OLD_GREEK_BUILD_RECEIPT" \
    --repo-root "$REPO_ROOT" \
    --output "$INPUT_RECEIPT"
