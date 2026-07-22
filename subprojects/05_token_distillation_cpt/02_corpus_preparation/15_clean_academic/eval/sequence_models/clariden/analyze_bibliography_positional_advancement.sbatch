#!/usr/bin/env bash
# Apply the predeclared paired-work advancement gate to P0/P1 train OOF output.
#SBATCH --job-name=bib_pos_gate
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/%x-%j.err
set -euo pipefail
umask 077
CODE_ROOT=${CODE_ROOT:?set immutable code bundle root}
BASE_TABLE_DIR=${BASE_TABLE_DIR:?set frozen entry table}
ROLE_TARGET_DIR=${ROLE_TARGET_DIR:?set reviewed role target view}
MODEL_DIR=${MODEL_DIR:?set completed P0/P1 ladder directory}
OUTPUT=${OUTPUT:?set a new advancement report path}
EXPECTED_COMMIT=${EXPECTED_COMMIT:?set exact code commit}
UENV=${UENV:-pytorch/v2.9.1:v2}
[[ ! -e "$OUTPUT" && ! -L "$OUTPUT" ]] || exit 90
[[ -f "$CODE_ROOT/bundle.receipt.json" && ! -L "$CODE_ROOT/bundle.receipt.json" ]] || exit 91
python3 - "$CODE_ROOT" "$EXPECTED_COMMIT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
receipt = json.loads((root / "bundle.receipt.json").read_text(encoding="utf-8"))
assert receipt["code_commit"] == sys.argv[2]
for relative, expected in receipt["files"].items():
    path = root / relative
    assert path.is_file() and not path.is_symlink(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
PY
export CUDA_VISIBLE_DEVICES="" NVIDIA_VISIBLE_DEVICES=void ROCR_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES=""
uenv run "$UENV" --view=default -- env PYTHONPATH="$CODE_ROOT" \
  python3 -m sequence_models.bibliography_positional_advancement \
    --base-table-dir "$BASE_TABLE_DIR" --role-target-dir "$ROLE_TARGET_DIR" \
    --model-dir "$MODEL_DIR" --output "$OUTPUT" \
    --precision-target 0.99 --maximum-source-recall-loss 0.01 \
    --bootstrap-replicates 2000 --seed 20260714 \
    --code-commit "$EXPECTED_COMMIT" --slurm-job-id "$SLURM_JOB_ID"
echo "BIBLIOGRAPHY_POSITIONAL_ADVANCEMENT=$OUTPUT"
