#!/usr/bin/env bash
#SBATCH --job-name=v2_split_replay_tok
#SBATCH --account=a0140
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/curriculum_v2/%x-%j.err
# Split the already-clean replay_only_final.jsonl into:
#   foreign_replay_only_final.jsonl
#   old_greek_replay_only_final.jsonl
# then tokenize each split for base/ext. CPU-only; no GPUs requested.
set -euo pipefail
V2_DIR="${V2_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
if [ ! -f "$V2_DIR/paths.env" ] && [ -f "$V2_DIR/../paths.env" ]; then
  V2_DIR="$(cd "$V2_DIR/.." && pwd)"
fi
source "$V2_DIR/paths.env"

INPUT="${INPUT:-$STAGE/replay_only_final.jsonl}"
FOREIGN_JSONL="${FOREIGN_JSONL:-$STAGE/foreign_replay_only_final.jsonl}"
OLD_GREEK_JSONL="${OLD_GREEK_JSONL:-$STAGE/old_greek_replay_only_final.jsonl}"
MANIFEST="${MANIFEST:-$STAGE/replay_split_for_lr_manifest.json}"
REUSE_SPLIT="${REUSE_SPLIT:-0}"

mkdir -p "$STAGE" "$MEGOUT"
test -s "$INPUT"
check_build_py

if [ "$REUSE_SPLIT" = "1" ]; then
  test -s "$FOREIGN_JSONL"
  test -s "$OLD_GREEK_JSONL"
  test -s "$MANIFEST"
  echo "reusing split replay JSONLs"
else
  rm -f "$FOREIGN_JSONL" "$OLD_GREEK_JSONL" "$MANIFEST"
  run_build_py -u "$V2/dataset/split_replay_final_for_lr.py" \
    --input "$INPUT" \
    --foreign-output "$FOREIGN_JSONL" \
    --old-greek-output "$OLD_GREEK_JSONL" \
    --manifest "$MANIFEST"
fi

run_build_py - <<PY
import json
from pathlib import Path
m = json.loads(Path("$MANIFEST").read_text())
assert m["counts"]["foreign_rows"] > 0, m["counts"]
assert m["counts"]["old_greek_rows"] > 0, m["counts"]
assert m["counts"]["missing_source_rows"] == 0, m["counts"]
print(json.dumps({
    "foreign_rows": m["counts"]["foreign_rows"],
    "old_greek_rows": m["counts"]["old_greek_rows"],
    "foreign_est_tokens": m["approx_tokens_by_chars_over_4"]["foreign"],
    "old_greek_est_tokens": m["approx_tokens_by_chars_over_4"]["old_greek"],
}, indent=2), flush=True)
PY

cd "$MEG"
for spec in \
  "foreign_replay_only $FOREIGN_JSONL" \
  "old_greek_replay_only $OLD_GREEK_JSONL"
do
  set -- $spec
  B="$1"; JSONL="$2"
  for pair in "base $BASE_TOK" "ext $EXT_TOK"; do
    set -- $pair
    TAG="$1"; TOK="$2"
    echo "=== tokenize $B $TAG ==="
    rm -f "$MEGOUT/${B}_${TAG}_text_document.bin" "$MEGOUT/${B}_${TAG}_text_document.idx"
    run_build_py tools/preprocess_data.py --input "$JSONL" \
      --output-prefix "$MEGOUT/${B}_${TAG}" \
      --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$TOK" \
      --append-eod --workers 64 --json-keys text
    test -s "$MEGOUT/${B}_${TAG}_text_document.bin"
    test -s "$MEGOUT/${B}_${TAG}_text_document.idx"
  done
done

echo "split replay tokenize DONE $(date -u)"
ls -lah "$FOREIGN_JSONL" "$OLD_GREEK_JSONL" "$MANIFEST" "$MEGOUT"/foreign_replay_only_* "$MEGOUT"/old_greek_replay_only_*
