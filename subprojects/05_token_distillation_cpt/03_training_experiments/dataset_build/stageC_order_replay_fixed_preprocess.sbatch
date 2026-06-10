#!/usr/bin/env bash
#SBATCH --job-name=cpt13b_C_order_preprocess
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.err
# Reorder only HPLT/openarchives slots while preserving replay positions, then
# tokenize the ordered shared text stream for both arms.
set -euo pipefail

SC=/iopsstor/scratch/cscs/fffoivos
SUB=$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt
MEG=$SC/code/training/Megatron-LM-Swiss-AI
STAGE=$SC/cpt_corpus/cpt_2arm_13b
BASE_TOK=$SC/models/apertus-8b-2509
EXT_TOK=$SC/tokenizers/apertus_greek_modern_only_148480
PY=$SC/python_envs/cpt_build_xfer_py312/bin/python

INPUT_FINAL="${INPUT_FINAL:-$STAGE/bulk_mix_final.jsonl}"
ORDERED_FINAL="${ORDERED_FINAL:-$STAGE/bulk_mix_ordered_replay_fixed_final.jsonl}"
ORDERED_MANIFEST="${ORDERED_MANIFEST:-$STAGE/bulk_mix_ordered_replay_fixed_manifest.json}"
REUSE_ORDERED_FINAL="${REUSE_ORDERED_FINAL:-0}"
BASE_PREFIX="${BASE_PREFIX:-$STAGE/megatron/bulk_mix_ordered_replay_fixed_base}"
EXT_PREFIX="${EXT_PREFIX:-$STAGE/megatron/bulk_mix_ordered_replay_fixed_ext}"

mkdir -p "$STAGE/megatron"
test -x "$PY"
test -s "$INPUT_FINAL"

echo "=== stageC xfer env preflight ==="
"$PY" - <<PY
import sys
import torch
import transformers
from transformers import AutoTokenizer
sys.path.insert(0, "$MEG")
from megatron.core.datasets import indexed_dataset
checks = [
    ("$BASE_TOK", 131072),
    ("$EXT_TOK", 148480),
]
for path, expected_vocab in checks:
    tok = AutoTokenizer.from_pretrained(path)
    actual = len(tok)
    print(f"tokenizer {path}: vocab={actual}")
    if actual != expected_vocab:
        raise SystemExit(f"unexpected vocab for {path}: {actual} != {expected_vocab}")
assert indexed_dataset is not None
print("stageC preflight OK")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
PY

if [ "$REUSE_ORDERED_FINAL" = "1" ]; then
  test -s "$ORDERED_FINAL"
  test -s "$ORDERED_MANIFEST"
  echo "=== reuse existing replay-fixed ordered final ==="
else
  rm -f "$ORDERED_FINAL" "$ORDERED_MANIFEST"
  rm -rf "$STAGE/.bulk_mix_ordered_replay_fixed_final.jsonl.parts"
  echo "=== reorder new-Greek slots, preserve replay positions ==="
  "$PY" -u "$SUB/03_training_experiments/dataset_build/reorder_new_greek_slots.py" \
    --input "$INPUT_FINAL" \
    --output "$ORDERED_FINAL" \
    --manifest "$ORDERED_MANIFEST" \
    --temp-dir "$STAGE/.bulk_mix_ordered_replay_fixed_final.jsonl.parts" \
    --progress-every 1000000
fi

"$PY" - <<PY
import json
from pathlib import Path
m = json.loads(Path("$ORDERED_MANIFEST").read_text())
assert m["replay_preservation"]["non_new_positions_preserved"], m["replay_preservation"]
assert m["new_greek_order"]["ordered_violation_count"] == 0, m["new_greek_order"]
assert m["counts"]["total_rows"] == m["counts"]["hplt_rows_written"] + m["counts"]["openarchives_rows_written"] + m["counts"]["non_new_greek_rows_written"]
print("ordered manifest gate OK")
print(json.dumps({
    "total_rows": m["counts"]["total_rows"],
    "hplt_rows": m["counts"]["hplt_rows"],
    "openarchives_rows": m["counts"]["openarchives_rows"],
    "non_new_greek_rows": m["counts"]["non_new_greek_rows"],
    "last_hplt_new_greek_slot": m["new_greek_order"]["last_hplt_new_greek_slot"],
    "first_openarchives_new_greek_slot": m["new_greek_order"]["first_openarchives_new_greek_slot"],
    "replay_positions_preserved": m["replay_preservation"]["non_new_positions_preserved"],
}, indent=2))
PY

rm -f "${BASE_PREFIX}_text_document.bin" "${BASE_PREFIX}_text_document.idx"
rm -f "${EXT_PREFIX}_text_document.bin" "${EXT_PREFIX}_text_document.idx"

echo "ordered docs: $(wc -l < "$ORDERED_FINAL")"
echo "=== tokenize ARM1 base 131072 ==="
cd "$MEG"
"$PY" tools/preprocess_data.py --input "$ORDERED_FINAL" \
  --output-prefix "$BASE_PREFIX" \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$BASE_TOK" \
  --append-eod --workers 64 --json-keys text
test -s "${BASE_PREFIX}_text_document.bin"
test -s "${BASE_PREFIX}_text_document.idx"

echo "=== tokenize ARM2 ext 148480 ==="
"$PY" tools/preprocess_data.py --input "$ORDERED_FINAL" \
  --output-prefix "$EXT_PREFIX" \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$EXT_TOK" \
  --append-eod --workers 64 --json-keys text
test -s "${EXT_PREFIX}_text_document.bin"
test -s "${EXT_PREFIX}_text_document.idx"

echo "stageC DONE $(date -u)"
ls -lah "${BASE_PREFIX}_text_document."* "${EXT_PREFIX}_text_document."*
