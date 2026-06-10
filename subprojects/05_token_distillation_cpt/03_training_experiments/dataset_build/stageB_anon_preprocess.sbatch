#!/usr/bin/env bash
#SBATCH --job-name=cpt13b_B_anon_preprocess
#SBATCH --account=a0140
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.out
#SBATCH --error=/capstor/scratch/cscs/fffoivos/runs/cpt_2arm_13b/%x-%j.err
# anonymize (email/IP/IBAN over the FULL mixed stream) -> tokenize x2 (base 131072 + ext 148480).
set -euo pipefail
SC=/iopsstor/scratch/cscs/fffoivos
SUB=$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt
MEG=$SC/code/training/Megatron-LM-Swiss-AI
STAGE=$SC/cpt_corpus/cpt_2arm_13b
BASE_TOK=$SC/models/apertus-8b-2509
EXT_TOK=$SC/tokenizers/apertus_greek_modern_only_148480
PY=$SC/python_envs/cpt_build_xfer_py312/bin/python
REUSE_FINAL="${REUSE_FINAL:-0}"
mkdir -p "$STAGE/decontam_indir" "$STAGE/megatron"
test -x "$PY"
echo "=== stageB xfer env preflight ==="
"$PY" - <<PY
import datatrove
import numpy
import orjson
import sys
import torch
import transformers
sys.path.insert(0, "$SUB/02_corpus_preparation/40_anonymize/scripts")
sys.path.insert(0, "$MEG")
from datatrove.data import Document
from megatron.core.datasets import indexed_dataset
from megatron.training.tokenizer import build_tokenizer
from pii_formatter import PIIFormatter
from transformers import AutoTokenizer

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
doc = Document(text="contact test@example.com", id="pii_smoke")
doc = list(PIIFormatter().run([doc]))[0]
assert doc.text == "contact <email-pii>", doc.text
assert doc.metadata["pii_count"] == 1, doc.metadata
assert indexed_dataset is not None
assert build_tokenizer is not None
print("stageB preflight OK")
print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("transformers", transformers.__version__)
PY
rm -f "$STAGE/megatron/bulk_mix_base_text_document.bin" "$STAGE/megatron/bulk_mix_base_text_document.idx"
rm -f "$STAGE/megatron/bulk_mix_ext_text_document.bin" "$STAGE/megatron/bulk_mix_ext_text_document.idx"
if [ "$REUSE_FINAL" = "1" ]; then
  test -s "$STAGE/bulk_mix_final.jsonl"
  echo "=== reuse existing anonymized final ==="
else
# datatrove shards by FILE: split the single decontam jsonl into 64 line-shards
# so all 64 tasks work (a single file would leave 63 cores idle).
  rm -f "$STAGE/decontam_indir"/part_*.jsonl
  rm -rf "$STAGE/masked"
  rm -f "$STAGE/bulk_mix_final.jsonl"
  split -n l/64 -d -a 2 --additional-suffix=.jsonl "$STAGE/bulk_mix_decontam.jsonl" "$STAGE/decontam_indir/part_"
  echo "=== anonymize (mask email/IP/IBAN over full mix incl. replay) ==="
  "$PY" -u "$SUB/02_corpus_preparation/40_anonymize/scripts/anonymize.py" \
    --input "$STAGE/decontam_indir" --reader jsonl --glob '*.jsonl' \
    --text-key text --id-key doc_id \
    --output "$STAGE/masked" --writer jsonl --tasks 64
  # datatrove JsonlWriter default output is gzipped ${rank}.jsonl.gz
  zcat "$STAGE"/masked/*.jsonl.gz > "$STAGE/bulk_mix_final.jsonl"
fi
echo "masked docs: $(wc -l < "$STAGE/bulk_mix_final.jsonl")"
echo "=== tokenize ARM1 base 131072 ==="
cd "$MEG"
"$PY" tools/preprocess_data.py --input "$STAGE/bulk_mix_final.jsonl" \
  --output-prefix "$STAGE/megatron/bulk_mix_base" \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$BASE_TOK" \
  --append-eod --workers 64 --json-keys text
test -s "$STAGE/megatron/bulk_mix_base_text_document.bin"
test -s "$STAGE/megatron/bulk_mix_base_text_document.idx"
echo "=== tokenize ARM2 ext 148480 ==="
"$PY" tools/preprocess_data.py --input "$STAGE/bulk_mix_final.jsonl" \
  --output-prefix "$STAGE/megatron/bulk_mix_ext" \
  --tokenizer-type HuggingFaceTokenizer --tokenizer-model "$EXT_TOK" \
  --append-eod --workers 64 --json-keys text
test -s "$STAGE/megatron/bulk_mix_ext_text_document.bin"
test -s "$STAGE/megatron/bulk_mix_ext_text_document.idx"
echo "stageB DONE $(date -u)"
ls -lah "$STAGE/megatron/"*text_document* 2>&1 | head
